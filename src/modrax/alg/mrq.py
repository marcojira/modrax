from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float
from pydantic import SkipValidation

from modrax.alg.base import Alg, AlgConfig
from modrax.network.base import Network, NetworkConfig
from modrax.network.block.linear import Linear, LinearConfig
from modrax.network.block.mlp import MLP, MLPConfig
from modrax.network.block_network import BlockNetworkConfig
from modrax.optimizer import Optimizer
from modrax.policy import PolicyFn
from modrax.rollout.base import RolloutConfig, RolloutFn
from modrax.types import Shape
from modrax.update.base import make_transition_minibatches, update_network


class MRQConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    rollout_fn: SkipValidation[RolloutFn]
    rollout_config: RolloutConfig

    policy_fn: SkipValidation[PolicyFn]

    buffer_size: int = 100_000
    minibatch_size: int = 512

    encoder_horizon: int = 32
    reward_horizon: int = 32

    discount: float = 0.99
    target_policy_noise: float = 0.2
    noise_clip: float = 0.3


class MRQNetworkConfig(NetworkConfig):
    encoder_cls: type[Network]
    encoder_config: BlockNetworkConfig

    action_dim: int
    state_action_encoder_config: MLPConfig
    state_action_dim: int

    value_config: MLPConfig
    policy_config: MLPConfig


class MRQNetwork(Network):
    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        config: MRQNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.num_actions = num_actions

        self.encoder = config.encoder_cls(obs_shape, num_actions, config.encoder_config, rngs)
        self.encoder_dim = config.encoder_config.heads["state_encoding"][-1]
        zsa_dim = self.encoder_dim + config.action_dim

        self.action_encoder = Linear(num_actions, config.action_dim, LinearConfig(), rngs)
        self.state_action_encoder = MLP(
            zsa_dim,
            config.state_action_dim,
            config.state_action_encoder_config,
            rngs,
        )
        self.mdp_predictor = Linear(
            config.state_action_dim, self.encoder_dim + 2, LinearConfig(), rngs
        )

        self.value_1 = MLP(config.state_action_dim, 1, config.value_config, rngs)
        self.value_2 = MLP(config.state_action_dim, 1, config.value_config, rngs)

        self.policy = MLP(self.encoder_dim, num_actions, config.policy_config, rngs)

    def __call__(self, inputs: dict[str, Float[Array, "B ..."]]) -> dict[str, Array]:
        z_s = self.encoder(inputs)["state_encoding"]

        return {"policy": self.policy(z_s), "z_s": z_s}

    def unroll_encoder(self, s_0, actions):
        zs_0 = self.encoder({"obs": s_0})["state_encoding"]

        def _step(carry, action):
            zs_curr = carry
            action = jax.nn.one_hot(action, self.num_actions)
            za = self.action_encoder(action)
            zsa_curr = self.state_action_encoder(jnp.concat([zs_curr, za], axis=-1))

            pred = self.mdp_predictor(zsa_curr)

            return pred[:, : self.encoder_dim], {
                "zs_pred": pred[:, : self.encoder_dim],
                "rewards_pred": pred[:, self.encoder_dim],
                "dones_pred": pred[:, self.encoder_dim + 1],
            }

        _, pred = nnx.scan(_step, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1))(zs_0, actions)
        return pred

    def value_target(self, zsa):
        return self.value_1(zsa), self.value_2(zsa)

    def state_encode(self, x):
        return self.encoder({"obs": x})["state_encoding"]

    def state_action_encode(self, zs, a):
        a = jax.nn.one_hot(a, self.num_actions, axis=-1)
        za = self.action_encoder(a)
        return self.state_action_encoder(jnp.concat([zs, za], axis=-1))

    def get_policy_horizon(self, obs_horizon):
        x = obs_horizon
        flat_x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        zs_horizon = self.encoder({"obs": flat_x})["state_encoding"]

        policy_logits = self.policy(zs_horizon)
        policy_logits = policy_logits.reshape(x.shape[0], x.shape[1], -1)
        zs_horizon = zs_horizon.reshape(x.shape[0], x.shape[1], -1)
        return zs_horizon, policy_logits

    def get_value_horizon(self, zs_horizon, action_horizon):
        x = zs_horizon
        flat_x = x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
        action_horizon = action_horizon.reshape(x.shape[0] * x.shape[1], *action_horizon.shape[2:])
        za = self.action_encoder(jax.nn.one_hot(action_horizon, self.num_actions, axis=-1))

        zsa_horizon = self.state_action_encoder(jnp.concat([flat_x, za], axis=-1))
        q_horizon = self.value_target(zsa_horizon)

        return jax.tree.map(lambda q: q.reshape(x.shape[0], x.shape[1], -1), q_horizon)


def make_window_array(x, window_size):
    H = window_size
    T = x.shape[1]

    indices = jnp.arange(T)[:, None] + jnp.arange(H)[None, :]
    y = x[:, indices, ...]
    return y


def encoder_update(network, optimizer, data, config, key):
    def encoder_loss(network, minibatch, config):
        pred = network.unroll_encoder(minibatch["obs"], minibatch["actions"])

        reward_loss = (minibatch["rewards"] - pred["rewards_pred"]) ** 2
        # dynamics_loss = minibatch[""]
        terminal_loss = (minibatch["dones"] - pred["dones_pred"]) ** 2

        loss = reward_loss + terminal_loss
        return loss.mean(), 0

    minibatches = make_transition_minibatches(data, key, config.minibatch_size)

    return update_network(network, optimizer, minibatches, encoder_loss, config)


def rl_update(network, optimizer, data, config, key):
    def rl_loss(network, minibatch, config):
        # Construct target
        zs_horizon, policy_logits = network.get_policy_horizon(minibatch["obs_horizon"])

        noise = jnp.clip(
            jax.random.normal(key, policy_logits.shape) * config.target_policy_noise
            - config.noise_clip,
            config.noise_clip,
        )
        next_action = jnp.argmax(policy_logits + noise, axis=-1)

        q_horizon = network.get_value_horizon(zs_horizon, next_action)
        q_horizon = jnp.minimum(q_horizon[0], q_horizon[1])
        q_horizon = q_horizon.squeeze(-1)

        target_reward_scale, reward_scale = 1, 1
        term_discount = config.discount**config.reward_horizon
        q_target = (
            minibatch["rewards"] + term_discount * q_horizon[:, -1] * target_reward_scale
        ) / reward_scale
        q_target = jax.lax.stop_gradient(q_target)

        # Get current
        zs = network.state_encode(minibatch["obs"])
        zsa = network.state_action_encode(zs, minibatch["action"])
        zs, zsa = jax.lax.stop_gradient(zs), jax.lax.stop_gradient(zsa)

        Q = network.value_target(zsa)

        value_loss = optax.huber_loss(Q[0].squeeze(-1) - q_target) + optax.huber_loss(
            Q[1].squeeze(-1) - q_target
        )

        policy_logits = network.policy(zs)
        action = jax.random.categorical(key, policy_logits)

        zsa = network.state_action_encode(zs, action)
        Q_policy = network.value_target(zsa)
        policy_loss = (
            -(Q_policy[0].squeeze(-1) + Q_policy[0].squeeze(-1)) / 2 + (policy_logits**2).mean()
        )

        loss = value_loss + policy_loss
        return loss.mean(), 0

    # Compute multi-step returns
    discounts = jnp.arange(config.reward_horizon)
    discounts = jnp.pow(config.discount * jnp.ones_like(discounts), discounts)
    rewards = data["rewards"] * discounts[None, :]
    data["rewards"] = rewards.sum(axis=-1)

    minibatches = make_transition_minibatches(data, key, config.minibatch_size)

    return update_network(network, optimizer, minibatches, rl_loss, config)


class MRQAlg(Alg):
    def __init__(
        self, alg_config: MRQConfig, env_state, recurrent_state, step_fn, jit: bool = False
    ) -> None:
        self.config = alg_config
        self.jit = jit
        self.step_fn = step_fn

        if jit:
            self.rollout_fn = nnx.jit(
                self.config.rollout_fn, static_argnames=["policy_fn", "step_fn", "config"]
            )
            self.encoder_update_fn = nnx.jit(encoder_update, static_argnames=["config"])
            self.rl_update_fn = nnx.jit(rl_update, static_argnames=["config"])

        else:
            self.rollout_fn = self.config.rollout_fn
            self.encoder_update_fn = encoder_update
            self.rl_update_fn = rl_update

        self.env_state = env_state
        self.recurrent_state = recurrent_state

    def __call__(self, network, optimizer, iteration_key) -> tuple[Network, Optimizer]:
        key, rollout_key, train_key = jax.random.split(iteration_key, 3)

        # Generate data
        env_state, recurrent_state, data = self.rollout_fn(
            network,
            self.config.policy_fn,
            self.step_fn,
            self.env_state,
            self.recurrent_state,
            self.config.rollout_config,
            rollout_key,
        )
        self.env_state, self.recurrent_state = env_state, recurrent_state

        # Add to buffer
        max_horizon_length = max(self.config.encoder_horizon, self.config.reward_horizon)
        obs_horizon = make_window_array(data.trajectory.obs, max_horizon_length)
        action_horizon = make_window_array(data.trajectory.actions, self.config.encoder_horizon)
        dones_horizon = make_window_array(data.trajectory.dones, self.config.encoder_horizon)
        rewards_horizon = make_window_array(data.trajectory.rewards, self.config.reward_horizon)

        all_data = (data, obs_horizon, action_horizon, dones_horizon, rewards_horizon)
        filtered_data = jax.tree.map(
            lambda x: x[:, : -(max_horizon_length - 1)], all_data
        )  # Remove transitions that don't have full horizon
        flat_data = jax.tree.map(lambda x: jnp.reshape(x, (-1, *x.shape[2:])), filtered_data)

        # print(flat_data[1].shape)

        # Train encoder
        encoder_data = {
            "obs": flat_data[0].trajectory.obs,
            "actions": flat_data[2],
            "dones": flat_data[3],
            "rewards": flat_data[4],
        }
        self.encoder_update_fn(network, optimizer, encoder_data, self.config, train_key)

        # Train RL
        rl_data = {
            "obs": flat_data[0].trajectory.obs,
            "action": flat_data[0].trajectory.actions,
            "obs_horizon": flat_data[1],
            "rewards": flat_data[4],
        }
        self.rl_update_fn(network, optimizer, rl_data, self.config, train_key)
        return network, optimizer
