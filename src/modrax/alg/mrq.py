import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from flax.struct import PyTreeNode
from jaxtyping import Array, Float, Key

from modrax.alg.base import Alg
from modrax.buffer import BufferState, ReplayBuffer
from modrax.env.base import Env, StateWithMetrics
from modrax.network.base import Network
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer
from modrax.rollout.trajectory_rollout import Trajectory, trajectory_rollout
from modrax.types import Config, Shape
from modrax.utils import finite_mean, update_network


@dataclass(frozen=True)
class MRQConfig(Config):
    num_envs: int = 32
    num_gen_steps: int = 256
    grad_steps: int = 4

    buffer_size: int = 1_000_000
    min_buffer_samples: int = 10_000
    minibatch_size: int = 512

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


@dataclass(frozen=True)
class MRQNetworkConfig(Config):
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

    # Policy
    exploration_noise: float = 0.1
    gumbel_tau: float = 10.0


class ValueNetwork(nnx.Module):
    """Single Q-network for MRQ."""

    def __init__(self, zsa_dim: int, hdim: int, activation: Callable, rngs: nnx.Rngs):
        self.q1 = MLP(zsa_dim, (hdim, hdim), hdim, activation, rngs, layer_norm=True)
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
            from modrax.network.cnn import CNN

            self.state_encoder = CNN(
                obs_shape,
                config.zs_dim,
                rngs,
                activation_fn=config.enc_activation,
            )
        else:
            input_dim = math.prod(obs_shape)
            self.state_encoder = MLP(
                input_dim,
                (config.enc_hdim, config.enc_hdim),
                config.zs_dim,
                config.enc_activation,
                rngs,
                layer_norm=True,
            )

        # Action encoder (za)
        self.action_encoder = nnx.Linear(num_actions, config.za_dim, rngs=rngs)

        self.state_action_encoder = MLP(
            config.zs_dim + config.za_dim,
            (config.enc_hdim, config.enc_hdim),
            config.zsa_dim,
            config.enc_activation,
            rngs,
            layer_norm=True,
        )  # State-action encoder (zsa)

        self.mdp_predictor = nnx.Linear(
            config.zsa_dim, 1 + config.zs_dim + config.num_bins, rngs=rngs
        )  # MDP predictor (next_zs, reward, done)

        # Value networks (twin Q-networks)
        self.value_1 = ValueNetwork(
            config.zsa_dim, config.value_hdim, config.value_activation, rngs
        )
        self.value_2 = ValueNetwork(
            config.zsa_dim, config.value_hdim, config.value_activation, rngs
        )

        # Actor network
        self.actor = MLP(
            config.zs_dim,
            (config.policy_hdim, config.policy_hdim),
            num_actions,
            config.policy_activation,
            rngs,
            layer_norm=True,
        )

    def policy(
        self, env_state: StateWithMetrics, key: Key[Array, ""]
    ) -> tuple[Float[Array, " B"], dict[str, Array]]:
        """Select action using gumbel softmax with optional exploration noise."""
        obs = env_state.obs
        action_mask = env_state.action_mask
        gumbel_key, noise_key = jax.random.split(key)

        z_s = self.state_encoder(obs)
        logits = self.actor(z_s)

        masked_logits = jnp.where(action_mask, logits, -jnp.inf)
        probs = gumbel_softmax(masked_logits, gumbel_key, self.config.gumbel_tau)

        noise = jax.random.normal(noise_key, probs.shape) * self.config.exploration_noise
        masked_probs = jnp.where(action_mask, probs + noise, -jnp.inf)
        action = masked_probs.argmax(axis=-1)

        return action, {"logits": logits, "z_s": z_s}

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


def gumbel_softmax(
    logits: Float[Array, "B A"], key: Key[Array, ""], tau: float = 1.0
) -> Float[Array, "B A"]:
    gumbel_noise = jax.random.gumbel(key, logits.shape)
    return jax.nn.softmax((logits + gumbel_noise) / tau, axis=-1)


def encoder_update(network, target_network, optimizer, batch, config, key):
    def encoder_loss(network, minibatch, config):
        encoder_target = target_network.state_encoder(
            jnp.reshape(minibatch.obs_horizon, (-1, *minibatch.obs_horizon.shape[2:]))
        )  # [B * H, D]
        encoder_target = jnp.reshape(encoder_target, (*minibatch.obs_horizon.shape[:2], -1))
        encoder_target = encoder_target[:, 1:]  # [B, H-1, D], predict the next state encoding

        pred = network.unroll_encoder(minibatch.obs, minibatch.action_horizon)

        dynamics_loss = ((encoder_target - pred["zs_pred"][:, :-1]) ** 2).mean(axis=-1)
        dynamics_loss = (dynamics_loss * minibatch.dones_mask[:, :-1]).sum(axis=1)

        reward_loss = (minibatch.rewards_horizon - pred["rewards_pred"]) ** 2
        reward_loss = (reward_loss * minibatch.terminal_mask).sum(axis=1)

        terminal_loss = (minibatch.dones_horizon - pred["dones_pred"]) ** 2
        terminal_loss = (terminal_loss * minibatch.terminal_mask).sum(axis=1)

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

    return update_network(network, optimizer, batch, encoder_loss, config)


def rl_update(network, target_network, optimizer, batch, config, key):
    target_reward_scale, reward_scale = 1, 1
    term_discount = config.discount**config.reward_horizon

    def rl_loss(network, minibatch, config):
        target_gumbel_key, target_noise_key, policy_gumbel_key = jax.random.split(key, 3)

        next_obs = minibatch.obs_horizon[:, config.reward_horizon]
        next_zs = target_network.state_encoder(next_obs)

        policy_logits = target_network.actor(next_zs)
        probs = gumbel_softmax(policy_logits, target_gumbel_key, network.config.gumbel_tau)
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
        next_q = next_q * minibatch.q_mask

        q_target = (
            minibatch.multi_step_reward + term_discount * next_q * target_reward_scale
        ) / reward_scale
        q_target = jax.lax.stop_gradient(q_target)

        # Get current
        zs = network.state_encoder(minibatch.obs)
        zsa = network.state_action_encode(
            zs, jax.nn.one_hot(minibatch.action, num_classes=network.num_actions)
        )
        zs, zsa = jax.lax.stop_gradient(zs), jax.lax.stop_gradient(zsa)

        # Value loss
        Q_0, Q_1 = network.values(zsa)
        value_loss = optax.huber_loss(Q_0.squeeze(-1), q_target)
        value_loss += optax.huber_loss(Q_1.squeeze(-1), q_target)

        # Policy loss
        policy_logits = network.actor(zs)
        probs = gumbel_softmax(policy_logits, policy_gumbel_key, network.config.gumbel_tau)
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

    return update_network(network, optimizer, batch, rl_loss, config)


class MRQSample(PyTreeNode):
    obs: Float[Array, "B ..."]
    action: Float[Array, "B ..."]
    obs_horizon: Float[Array, "B H ..."]
    action_horizon: Float[Array, "B H ..."]
    dones_horizon: Float[Array, "B H"]
    rewards_horizon: Float[Array, "B H"]
    dones_mask: Float[Array, "B H"]
    terminal_mask: Float[Array, "B H"]
    q_mask: Float[Array, " B"]
    multi_step_reward: Float[Array, " B"]


def pop(trajectory: Trajectory) -> Trajectory:
    """Remove the first timestep from a (B, T, ...) trajectory."""
    return jax.tree.map(lambda x: x[:, 1:], trajectory)


def add(trajectory: Trajectory, new: Trajectory) -> Trajectory:
    """Append a trajectory to the end along the time axis."""
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=1), trajectory, new)


@nnx.jit(static_argnames=["config"])
def make_sample_from_trajectory(trajectory: Trajectory, config: MRQConfig) -> MRQSample:
    """Create a sample for the first timestep from a trajectory (B, T, ...).

    Uses a window of max(encoder_horizon, reward_horizon) steps starting at t=0.
    """
    h = max(config.encoder_horizon, config.reward_horizon)
    obs_horizon = trajectory.obs[:, :h]
    action_horizon = trajectory.actions[:, :h]
    dones_horizon = trajectory.dones[:, :h]
    rewards_horizon = trajectory.rewards[:, :h]

    cumsum = jnp.cumsum(dones_horizon, axis=-1)
    shifted_cumsum = jnp.concatenate([jnp.zeros_like(cumsum[..., :1]), cumsum[..., :-1]], axis=-1)

    dones_mask = 1 - jnp.clip(cumsum, 0, 1)
    terminal_mask = 1 - jnp.clip(shifted_cumsum, 0, 1)
    q_mask = 1 - jnp.any(dones_horizon, axis=1).astype(jnp.float32)

    discounts = jnp.arange(config.reward_horizon)
    discounts = jnp.pow(config.discount * jnp.ones_like(discounts), discounts)
    rewards = rewards_horizon * terminal_mask
    multi_step_reward = (rewards[:, : config.reward_horizon] * discounts).sum(axis=-1)

    return MRQSample(
        obs=trajectory.obs[:, 0],
        action=trajectory.actions[:, 0],
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dones_horizon=dones_horizon,
        rewards_horizon=rewards_horizon,
        dones_mask=dones_mask,
        terminal_mask=terminal_mask,
        q_mask=q_mask,
        multi_step_reward=multi_step_reward,
    )


class MRQState(PyTreeNode):
    agent_state: Any
    env_state: StateWithMetrics
    buffer_state: BufferState
    trajectory: Trajectory


class MRQAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: MRQNetwork,
        optimizer: Optimizer,
        alg_cfg: MRQConfig,
        key: Key[Array, ""],
    ):
        self.env = env
        self.cfg = alg_cfg
        self.max_h = max(alg_cfg.encoder_horizon, alg_cfg.reward_horizon)

        # Setup
        network.eval()
        target_network = nnx.clone(network)

        self.env_steps_per_epoch = alg_cfg.num_envs * alg_cfg.num_gen_steps
        self.loop = nnx.scan(nnx.jit(self._loop))
        self.encoder_loop = nnx.scan(nnx.jit(self._encoder_step))
        self.gen_sample = nnx.jit(self._gen_sample, static_argnames=["random_action"])

        # Generate trajectories for buffer initialization
        env_state = self.env.reset(jax.random.split(key, alg_cfg.num_envs))
        num_steps = self.max_h
        env_state, trajectory = trajectory_rollout(
            network, self.env.step, env_state, num_steps, key
        )

        # Initialize buffer
        self.buffer = ReplayBuffer(max_size=alg_cfg.buffer_size)
        sample = make_sample_from_trajectory(trajectory, self.cfg)
        buffer_state = self.buffer.init(sample)

        # Initial fill of buffer
        num_steps = int(self.cfg.min_buffer_samples / alg_cfg.num_envs)
        for _ in range(num_steps):
            env_state, trajectory, buffer_state, _ = self.gen_sample(
                network, env_state, trajectory, buffer_state, key, random_action=True
            )

        self.state = MRQState(
            nnx.split((network, optimizer, target_network)),
            env_state,
            buffer_state,
            trajectory,
        )

    def _loop(self, state: MRQState, key: Key[Array, ""]):
        env_state, buffer_state, trajectory = state.env_state, state.buffer_state, state.trajectory
        network, optimizer, target_network = nnx.merge(*state.agent_state)
        rollout_key, train_key = jax.random.split(key)
        metrics = {}

        # Gen data
        new_env_state, trajectory, buffer_state, step_traj = self._gen_sample(
            network, env_state, trajectory, buffer_state, rollout_key
        )

        # Clear history for terminated envs (fill dones with 1 to mask future samples)
        # #(maybe this is still necessary but not sure? I've removed for now)

        # Record trajectory metrics
        num_dones = step_traj.dones.sum()
        episode_returns = (step_traj.episode_returns * step_traj.dones).sum() / num_dones
        episode_lengths = (step_traj.episode_lengths * step_traj.dones).sum() / num_dones

        metrics["return"] = jnp.where(num_dones > 0, episode_returns, -jnp.inf)
        metrics["lengths"] = jnp.where(num_dones > 0, episode_lengths, -jnp.inf)

        # Train
        for _ in range(self.cfg.grad_steps):
            train_key, rl_sample_key, rl_key = jax.random.split(train_key, 3)

            batch, _, _ = self.buffer.sample(buffer_state, rl_sample_key, self.cfg.minibatch_size)
            _, rl_info = rl_update(network, target_network, optimizer, batch, self.cfg, rl_key)

        metrics.update(jax.tree.map(lambda x: x.mean(), rl_info))

        return MRQState(
            nnx.split((network, optimizer, target_network)),
            new_env_state,
            buffer_state,
            trajectory,
        ), metrics

    def _gen_sample(self, network, env_state, trajectory, buffer_state, key, random_action=False):
        """Collect one step, update sliding trajectory window, and add to buffer."""
        env_state, step_traj = trajectory_rollout(
            network, self.env.step, env_state, 1, key, random_action=random_action
        )
        trajectory = pop(trajectory)
        trajectory = add(trajectory, step_traj)
        sample = make_sample_from_trajectory(trajectory, self.cfg)
        buffer_state = self.buffer.add(buffer_state, sample)
        return env_state, trajectory, buffer_state, step_traj

    def _encoder_step(self, state: MRQState, key: Key[Array, ""]):
        """Single encoder gradient update step."""
        network, optimizer, target_network = nnx.merge(*state.agent_state)
        sample_key, enc_key = jax.random.split(key)
        batch, _, _ = self.buffer.sample(state.buffer_state, sample_key, self.cfg.minibatch_size)
        _, info = encoder_update(network, target_network, optimizer, batch, self.cfg, enc_key)
        return MRQState(
            nnx.split((network, optimizer, target_network)),
            state.env_state,
            state.buffer_state,
            state.trajectory,
        ), info

    def __call__(self, key: Key[Array, ""]):
        # Update target network
        network, optimizer, target_network = nnx.merge(*self.state.agent_state)
        nnx.update(target_network, nnx.state(network))
        self.state = self.state.replace(agent_state=nnx.split((network, optimizer, target_network)))

        enc_key, loop_key = jax.random.split(key)

        # Update encoder
        enc_keys = jax.random.split(enc_key, self.cfg.num_gen_steps * self.cfg.grad_steps)
        self.state, enc_info = self.encoder_loop(self.state, enc_keys)

        # Gen data and update RL
        loop_keys = jax.random.split(loop_key, self.cfg.num_gen_steps)
        self.state, metrics = self.loop(self.state, loop_keys)
        metrics = {k: finite_mean(v) for k, v in metrics.items()}
        metrics.update(jax.tree.map(lambda x: x.mean(), enc_info))

        return metrics
