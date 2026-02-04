import math
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.base import Alg, AlgConfig
from modrax.buffer import BufferState, ReplayBuffer
from modrax.env.base import Env, EnvState
from modrax.network.base import Network, NetworkConfig
from modrax.network.block.base import RecurrentState
from modrax.network.module.mlp import MLP
from modrax.optimizer import Optimizer
from modrax.policy import uniform_policy
from modrax.rollout.base import RolloutData
from modrax.rollout.rollout import rollout
from modrax.types import Shape
from modrax.utils import compute_training_metrics, update_network


class MRQConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    num_gen_steps: int = 128

    buffer_size: int = 1_000_000
    min_buffer_samples: int = 10_000
    minibatch_size: int = 256

    encoder_horizon: int = 4
    reward_horizon: int = 4

    discount: float = 0.99
    target_policy_noise: float = 0.1
    noise_clip: float = 0.15

    # Encoder loss weights
    dyn_weight: float = 1.0
    reward_weight: float = 0.1
    done_weight: float = 0.1

    # Policy loss
    pre_activ_weight: float = 1e-5
    gumbel_tau: float = 10.0

    # Exploration
    exploration_noise: float = 0.1


class MRQNetworkConfig(NetworkConfig):
    model_config = {"arbitrary_types_allowed": True}

    pixel_obs: bool = False

    # Dimensions
    zs_dim: int = 512
    za_dim: int = 256
    zsa_dim: int = 512

    # MLP hidden dimensions
    enc_hdim: int = 512
    value_hdim: int = 512
    policy_hdim: int = 512

    # Activations
    enc_activation: Callable = jax.nn.elu
    value_activation: Callable = jax.nn.elu
    policy_activation: Callable = jax.nn.relu

    # Reward prediction
    num_bins: int = 65


class ValueNetwork(nnx.Module):
    """Single Q-network for MRQ."""

    def __init__(self, zsa_dim: int, hdim: int, activation: Callable, rngs: nnx.Rngs):
        from modrax.network.module.mlp import MLP

        self.q1 = MLP(zsa_dim, (hdim, hdim), hdim, activation, rngs, use_layer_norm=True)
        self.ln = nnx.LayerNorm(hdim, use_scale=False, use_bias=False, rngs=rngs)
        self.activation = activation
        self.q2 = nnx.Linear(hdim, 1, rngs=rngs)

    def __call__(self, x: Float[Array, "B D"]) -> Float[Array, "B 1"]:
        x = self.q1(x)
        x = self.ln(x)
        x = self.activation(x)
        return self.q2(x)


class MRQNetwork(Network):
    """MRQ Network matching PyTorch architecture."""

    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        config: MRQNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.num_actions = num_actions
        self.rngs = rngs

        # State encoder (zs)
        if config.pixel_obs:
            from modrax.network.block.cnn import CNN, CNNConfig

            self.state_encoder = CNN(
                obs_shape,
                config.zs_dim,
                CNNConfig(activation_fn=config.enc_activation),
                rngs,
            )
        else:
            input_dim = math.prod(obs_shape) if isinstance(obs_shape, tuple) else obs_shape
            self.state_encoder = MLP(
                input_dim,
                (config.enc_hdim, config.enc_hdim),
                config.zs_dim,
                config.enc_activation,
                rngs,
                use_layer_norm=True,
            )

        # Action encoder (za)
        self.action_encoder = nnx.Linear(num_actions, config.za_dim, rngs=rngs)

        # State-action encoder (zsa)
        self.state_action_encoder = MLP(
            config.zs_dim + config.za_dim,
            (config.enc_hdim, config.enc_hdim),
            config.zsa_dim,
            config.enc_activation,
            rngs,
            use_layer_norm=True,
        )

        # MDP predictor (next_zs, reward, done)
        self.mdp_predictor = nnx.Linear(
            config.zsa_dim, 1 + config.zs_dim + config.num_bins, rngs=rngs
        )

        # Value networks (twin Q-networks)
        self.value_1 = ValueNetwork(
            config.zsa_dim, config.value_hdim, config.value_activation, rngs
        )
        self.value_2 = ValueNetwork(
            config.zsa_dim, config.value_hdim, config.value_activation, rngs
        )

        # Policy network
        self.policy = MLP(
            config.zs_dim,
            (config.policy_hdim, config.policy_hdim),
            num_actions,
            config.policy_activation,
            rngs,
            use_layer_norm=True,
        )

    def __call__(self, inputs: dict[str, Float[Array, "B ..."]]) -> dict[str, Array]:
        z_s = self.state_encoder(inputs["obs"])

        return {"policy": self.policy(z_s), "z_s": z_s}

    def unroll_encoder(self, s_0, actions):
        def _step(carry, action):
            zs_curr = carry
            one_hot_action = jax.nn.one_hot(action, self.num_actions)
            zsa_curr = self.state_action_encode(zs_curr, one_hot_action)

            pred = self.mdp_predictor(zsa_curr)

            return pred[:, : self.config.zs_dim], {
                "zs_pred": pred[:, : self.config.zs_dim],
                "rewards_pred": pred[:, self.config.zs_dim],
                "dones_pred": pred[:, self.config.zs_dim + 1],
            }

        zs_0 = self.state_encoder(s_0)
        _, pred = nnx.scan(_step, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1))(zs_0, actions)
        return pred

    def values(self, zsa):
        return self.value_1(zsa), self.value_2(zsa)

    def state_action_encode(self, zs, a):
        za = self.config.enc_activation(self.action_encoder(a))
        return self.state_action_encoder(jnp.concat([zs, za], axis=-1))


def make_window_array(x, window_size):
    H = window_size
    T = x.shape[1]

    indices = jnp.arange(T)[:, None] + jnp.arange(H)[None, :]
    y = x[:, indices, ...]
    return y


def gumbel_softmax(
    logits: Float[Array, "B A"], key: Key[Array, ""], tau: float = 1.0
) -> Float[Array, "B A"]:
    gumbel_noise = jax.random.gumbel(key, logits.shape)
    return jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)


def mrq_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
    exploration_noise: float = 0.0,
    gumbel_tau: float = 10.0,
) -> Float[Array, " B"]:
    """MRQ action selection with optional exploration noise."""
    masked_logits = jnp.where(action_mask, logits, -jnp.inf)
    gumbel_key, noise_key = jax.random.split(key)
    probs = gumbel_softmax(masked_logits, gumbel_key, gumbel_tau)

    noise = jax.random.normal(noise_key, probs.shape) * exploration_noise
    masked_probs = jnp.where(action_mask, probs + noise, -jnp.inf)

    return (masked_probs).argmax(axis=-1)


def encoder_update(network, target_network, optimizer, batch, config, key):
    def encoder_loss(network, minibatch, config):
        encoder_target = target_network.state_encoder(
            jnp.reshape(minibatch["obs_horizon"], (-1, *minibatch["obs_horizon"].shape[2:]))
        )  # [B * H, D]
        encoder_target = jnp.reshape(encoder_target, (*minibatch["obs_horizon"].shape[:2], -1))
        encoder_target = encoder_target[:, 1:]  # [B, H-1, D], predict the next state encoding

        pred = network.unroll_encoder(minibatch["obs"], minibatch["action_horizon"])

        dynamics_loss = ((encoder_target - pred["zs_pred"][:, :-1]) ** 2).mean(axis=-1)
        dynamics_loss = (dynamics_loss * minibatch["dones_mask"][:, :-1]).sum(axis=1)

        reward_loss = (minibatch["rewards_horizon"] - pred["rewards_pred"]) ** 2
        reward_loss = (reward_loss * minibatch["terminal_mask"]).sum(axis=1)

        terminal_loss = (minibatch["dones_horizon"] - pred["dones_pred"]) ** 2
        terminal_loss = (terminal_loss * minibatch["terminal_mask"]).sum(axis=1)

        loss = (
            config.dyn_weight * dynamics_loss
            + config.reward_weight * reward_loss
            + config.done_weight * terminal_loss
        )

        info = {
            "dyn_loss": dynamics_loss.mean(),
            "rew_loss": reward_loss.mean(),
            "done_loss": terminal_loss.mean(),
        }
        return loss.mean(), info

    minibatches = jax.tree.map(lambda x: x.reshape(-1, config.minibatch_size, *x.shape[1:]), batch)

    return update_network(network, optimizer, minibatches, encoder_loss, config)


def rl_update(network, target_network, optimizer, batch, config, key):
    target_reward_scale, reward_scale = 1, 1
    term_discount = config.discount**config.reward_horizon

    def rl_loss(network, minibatch, config):
        target_gumbel_key, target_noise_key, policy_gumbel_key = jax.random.split(key, 3)

        next_obs = minibatch["obs_horizon"][:, config.reward_horizon]
        next_zs = target_network.state_encoder(next_obs)

        policy_logits = target_network.policy(next_zs)
        probs = gumbel_softmax(policy_logits, target_gumbel_key, config.gumbel_tau)
        noise = jnp.clip(
            jax.random.normal(target_noise_key, probs.shape) * config.target_policy_noise,
            -config.noise_clip,
            config.noise_clip,
        )
        next_action = jnp.argmax(probs + noise, axis=-1)
        next_action = jax.nn.one_hot(next_action, num_classes=network.num_actions)

        next_zsa = target_network.state_action_encode(next_zs, next_action)
        next_q1, next_q2 = target_network.values(next_zsa)
        next_q = jnp.minimum(next_q1, next_q2).squeeze(-1)

        # Mask out q if done was hit within reward horizon
        next_q = next_q * minibatch["q_mask"]

        q_target = (
            minibatch["multi_step_reward"] + term_discount * next_q * target_reward_scale
        ) / reward_scale
        q_target = jax.lax.stop_gradient(q_target)

        # Get current
        zs = network.state_encoder(minibatch["obs"])
        zsa = network.state_action_encode(
            zs, jax.nn.one_hot(minibatch["action"], num_classes=network.num_actions)
        )
        zs, zsa = jax.lax.stop_gradient(zs), jax.lax.stop_gradient(zsa)

        # Value loss
        Q_0, Q_1 = network.values(zsa)
        value_loss = optax.huber_loss(Q_0.squeeze(-1), q_target)
        value_loss += optax.huber_loss(Q_1.squeeze(-1), q_target)

        # Policy loss
        policy_logits = network.policy(zs)
        probs = gumbel_softmax(policy_logits, policy_gumbel_key, config.gumbel_tau)
        zsa_policy = target_network.state_action_encode(zs, probs)
        Q_0, Q_1 = target_network.values(
            zsa_policy
        )  # Technically should use the network itself while no gradient through value network
        policy_loss = -(Q_0.squeeze(-1) + Q_1.squeeze(-1)) / 2
        policy_loss += config.pre_activ_weight * (policy_logits**2).mean()

        loss = value_loss + policy_loss
        info = {
            "val_loss": value_loss.mean(),
            "policy_loss": policy_loss.mean(),
        }
        return loss.mean(), info

    minibatches = jax.tree.map(
        lambda x: x.reshape(-1, config.minibatch_size, *x.shape[1:]),
        batch,
    )

    return update_network(network, optimizer, minibatches, rl_loss, config)


@nnx.jit(static_argnames=["config"])
def make_samples(data: RolloutData, config):
    max_h = max(config.encoder_horizon, config.reward_horizon)
    t = data.trajectory

    sample_data = {
        "obs": t.obs,
        "action": t.actions,
        "obs_horizon": make_window_array(t.obs, max_h),
        "action_horizon": make_window_array(t.actions, max_h),
        "dones_horizon": make_window_array(t.dones, max_h),
        "rewards_horizon": make_window_array(t.rewards, max_h),
    }
    sample_data = jax.tree.map(lambda x: x[:, :-max_h], sample_data)  # TODO: Better filtering
    sample_data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sample_data)

    # Compute done masks
    cumsum = jnp.cumsum(sample_data["dones_horizon"], axis=-1)
    shifted_cumsum = jnp.concatenate([jnp.zeros_like(cumsum[..., :1]), cumsum[..., :-1]], axis=-1)

    # Mask done + states after
    sample_data["dones_mask"] = 1 - jnp.clip(cumsum, 0, 1)
    # Mask states after done
    sample_data["terminal_mask"] = 1 - jnp.clip(shifted_cumsum, 0, 1)
    # Mask q_value if done was hit before end
    sample_data["q_mask"] = 1 - jnp.any(sample_data["dones_horizon"], axis=1).astype(jnp.float32)

    # Compute multi-step returns
    discounts = jnp.arange(config.reward_horizon)
    discounts = jnp.pow(config.discount * jnp.ones_like(discounts), discounts)

    rewards = sample_data["rewards_horizon"] * sample_data["terminal_mask"]
    rewards = rewards[:, : config.reward_horizon] * discounts
    sample_data["multi_step_reward"] = rewards.sum(axis=-1)

    return sample_data


class MRQAlg(Alg):
    def __init__(
        self,
        env_state: EnvState,
        recurrent_state: RecurrentState | None,
        network: Network,
        optimizer: Optimizer,
        env: Env,
        alg_config: MRQConfig,
        jit: bool = False,
    ) -> None:
        self.config = alg_config
        self.jit = jit
        self.step_fn = env.step
        self.env = env

        if jit:
            self.rollout_fn = nnx.jit(
                rollout, static_argnames=["policy_fn", "step_fn", "num_steps"]
            )
            self.encoder_update_fn = nnx.jit(encoder_update, static_argnames=["config"])
            self.rl_update_fn = nnx.jit(rl_update, static_argnames=["config"])

        else:
            self.rollout_fn = rollout
            self.encoder_update_fn = encoder_update
            self.rl_update_fn = rl_update

        self.env_state = env_state
        self.recurrent_state = recurrent_state

        self.buffer = ReplayBuffer(max_size=alg_config.buffer_size, jit=jit)
        self.target_network = nnx.clone(network)

        # Initial fill of buffer
        key = jax.random.key(0)
        num_steps = int(self.config.min_buffer_samples / 32)
        env_state = self.env.reset(jax.random.split(key, 32))
        _, _, data = self.rollout_fn(
            network,
            uniform_policy,
            self.step_fn,
            env_state,
            self.recurrent_state,
            num_steps,
            jax.random.key(0),
        )
        sample_data = make_samples(data, self.config)

        self.buffer_state = self.buffer.init(sample_data)
        self.buffer_state = self.buffer.add(self.buffer_state, sample_data)

        self.policy_fn = partial(
            mrq_policy,
            exploration_noise=self.config.exploration_noise,
            gumbel_tau=self.config.gumbel_tau,
        )

        self.iteration = 0

    def __call__(self, network, optimizer, iteration_key):
        self.iteration += 1
        rollout_key, encoder_key, rl_key = jax.random.split(iteration_key, 3)

        """ GENERATE """
        env_state, recurrent_state, data = self.rollout_fn(
            network,
            self.policy_fn,
            self.step_fn,
            self.env_state,
            self.recurrent_state,
            self.config.num_gen_steps,
            rollout_key,
        )
        self.env_state, self.recurrent_state = env_state, recurrent_state

        # Add to buffer
        sample_data = make_samples(data, self.config)
        self.buffer_state = self.buffer.add(self.buffer_state, sample_data)

        # # Compute reward_scale
        # reward_scale = jnp.abs(self.buffer_state.data["encoder_rewards"][:, 0])
        # reward_scale = (reward_scale.sum() / self.buffer_state.size).clip(min=1e-8)

        """ TRAIN """
        # Update target network periodically
        nnx.update(self.target_network, nnx.state(network))

        minibatch_size, num_steps = self.config.minibatch_size, self.config.num_gen_steps

        # Encoder
        batch, _, _ = self.buffer.sample(self.buffer_state, encoder_key, minibatch_size * num_steps)
        _, encoder_info = self.encoder_update_fn(
            network, self.target_network, optimizer, batch, self.config, encoder_key
        )

        # RL
        batch, _, _ = self.buffer.sample(self.buffer_state, rl_key, minibatch_size * num_steps)
        _, rl_info = self.rl_update_fn(
            network, self.target_network, optimizer, batch, self.config, rl_key
        )

        metrics = compute_training_metrics(data.trajectory)
        metrics.update(jax.tree.map(lambda x: x.mean(), encoder_info))
        metrics.update(jax.tree.map(lambda x: x.mean(), rl_info))

        return (network, optimizer, metrics)
