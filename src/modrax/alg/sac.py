import math
from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float, Key

from modrax.alg.base import Alg, AlgConfig, OptimizerConfig, create_optimizer
from modrax.buffer import BufferState, ReplayBuffer
from modrax.env.base import ContinuousActionSpec, Env, StateWithMetrics
from modrax.metrics import finite_mean
from modrax.network.base import Network, NetworkConfig
from modrax.network.running_norm import RunningNorm
from modrax.rollout import Transition, trajectory_rollout, trajectory_to_transitions


@dataclass(frozen=True)
class SACConfig(AlgConfig):
    critic_optimizer_cfg: OptimizerConfig = OptimizerConfig(learning_rate=1e-3)
    actor_optimizer_cfg: OptimizerConfig = OptimizerConfig(learning_rate=1e-3)
    alpha_optimizer_cfg: OptimizerConfig = OptimizerConfig(learning_rate=3e-4)

    init_buffer_size: int = int(5e3)
    buffer_size: int = int(1e6)

    gamma: float = 0.99
    tau: float = 0.005

    num_envs: int = 128
    minibatch_size: int = 512
    grad_steps: int = 8
    iterations_per_epoch: int = 1024

    total_steps: int = 10_000_000

    autotune: bool = True


""" NETWORK """


@dataclass(frozen=True)
class SACNetworkConfig(NetworkConfig):
    use_running_norm: bool = True
    min_std: float = 0.001
    init_alpha: float = 1.0


class Actor(Network):
    def get_action(self, obs: Float[Array, "B D"], key: Key[Array, ""]) -> tuple:
        raise NotImplementedError


class Critic(Network):
    def get_qs(
        self, obs: Float[Array, "B D"], action: Float[Array, "B A"]
    ) -> tuple[Float[Array, " B"], Float[Array, " B"]]:
        raise NotImplementedError


class LogAlpha(Network):
    def __init__(self, value: float):
        self.param = nnx.Param(value * jnp.ones((1,)))  # Somehow prevents recompilation?


class SACNetwork(Network):
    """Abstract base class for SAC networks."""

    actor: Actor
    critic: Critic
    critic_target: Critic
    log_alpha: LogAlpha
    running_norm: Callable

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]) -> tuple:
        raise NotImplementedError


""" ALGORITHM """


def value_update(
    network: SACNetwork,
    optimizer: nnx.Optimizer,
    minibatch: Transition,
    cfg: SACConfig,
    key: Key[Array, ""],
):
    def value_loss(critic: Critic, batch: Transition, key: Key[Array, ""]):
        obs, next_obs, actions, rewards, dones, truncations = (
            network.running_norm(batch.obs),
            network.running_norm(batch.next_obs),
            batch.actions,
            batch.rewards,
            batch.dones,
            batch.truncations,
        )

        next_action, next_action_log_prob = network.actor.get_action(next_obs, key)

        next_action, next_action_log_prob = (
            jax.lax.stop_gradient(next_action),
            jax.lax.stop_gradient(next_action_log_prob),
        )

        q1, q2 = network.critic_target.get_qs(next_obs, next_action)
        min_q = jnp.minimum(q1, q2)
        min_q_next_target = min_q - jnp.exp(network.log_alpha.param[...]) * next_action_log_prob
        next_q_value = rewards + (1 - dones) * cfg.gamma * min_q_next_target

        q1, q2 = critic.get_qs(obs, actions)

        value_loss = jnp.square(q1 - next_q_value) + jnp.square(q2 - next_q_value)
        value_loss = value_loss * (1 - truncations)  # Zero out TD error for truncated transitions
        value_loss = 0.5 * jnp.mean(value_loss)
        return value_loss, {}

    network.critic.train()
    (loss, info), grads = nnx.value_and_grad(value_loss, has_aux=True)(
        network.critic, minibatch, key
    )
    optimizer.update(network.critic, grads)
    network.critic.eval()

    return loss, info


def policy_update(
    network: SACNetwork,
    optimizer: nnx.Optimizer,
    minibatch: Transition,
    cfg: SACConfig,
    key: Key[Array, ""],
):
    def policy_loss(actor: Actor, batch: Transition, key: Key[Array, ""]):
        obs = network.running_norm(batch.obs)
        sample_act, log_pi = actor.get_action(obs, key)
        qf1_act, qf2_act = network.critic.get_qs(obs, sample_act)
        min_qf_act = jnp.minimum(qf1_act, qf2_act)
        actor_loss = ((jnp.exp(network.log_alpha.param[...]) * log_pi) - min_qf_act).mean()
        return actor_loss, {}

    network.actor.train()
    (loss, info), grads = nnx.value_and_grad(policy_loss, has_aux=True)(
        network.actor, minibatch, key
    )
    optimizer.update(network.actor, grads)
    network.actor.eval()

    return loss, info


def alpha_update(
    network: SACNetwork,
    optimizer: nnx.Optimizer,
    minibatch: Transition,
    target_entropy: float,
    cfg: SACConfig,
    key: Key[Array, ""],
):
    def alpha_loss(log_alpha: LogAlpha, batch: Transition, key: Key[Array, ""]):
        obs = network.running_norm(batch.obs)
        _, log_pi = network.actor.get_action(obs, key)
        log_pi = jax.lax.stop_gradient(log_pi)
        alpha_loss = (-jnp.exp(log_alpha.param[...]) * (log_pi + target_entropy)).mean()
        return alpha_loss, {}

    (loss, info), grads = nnx.value_and_grad(alpha_loss, has_aux=True)(
        network.log_alpha, minibatch, key
    )
    optimizer.update(network.log_alpha, grads)

    return loss, info


def ema_update(source: nnx.Module, target: nnx.Module, tau: float):
    """Update target parameters with an exponential moving average of source parameters."""
    source_params = nnx.state(source, nnx.Param)
    target_params = nnx.state(target, nnx.Param)
    new_target_params = jax.tree.map(
        lambda source, target: tau * source + (1 - tau) * target,
        source_params,
        target_params,
    )
    nnx.update(target, new_target_params)

    source_batch_stats = nnx.state(source, nnx.BatchStat)
    nnx.update(target, source_batch_stats)


@struct.dataclass
class SACState:
    agent_state: Any
    env_state: StateWithMetrics
    buffer_state: BufferState


class SACAlg(Alg):
    cfg: SACConfig

    def __init__(
        self,
        env: Env,
        network: SACNetwork,
        alg_cfg: SACConfig,
        key: Key[Array, ""],
    ):
        if not isinstance(env.action_spec, ContinuousActionSpec):
            raise ValueError("SAC requires a continuous action space")

        super().__init__(env, network, alg_cfg)

        network.eval()
        self.target_entropy = -0.5 * math.prod(env.action_spec.shape)

        critic_optimizer = create_optimizer(network.critic, alg_cfg.critic_optimizer_cfg)
        actor_optimizer = create_optimizer(network.actor, alg_cfg.actor_optimizer_cfg)
        alpha_optimizer = create_optimizer(network.log_alpha, alg_cfg.alpha_optimizer_cfg)

        # Generate transitions for initial buffer
        reset_key, rollout_key = jax.random.split(key)
        env_state = self.env.reset(jax.random.split(reset_key, self.cfg.num_envs))
        num_steps = int(self.cfg.init_buffer_size / self.cfg.num_envs)

        env_state, trajectory = trajectory_rollout(network, self.env.step, env_state, num_steps, rollout_key)
        transitions = trajectory_to_transitions(trajectory, env_state)
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)

        # Init buffer
        self.buffer = ReplayBuffer(max_size=alg_cfg.buffer_size)
        buffer_state = self.buffer.init(transitions)
        buffer_state = self.buffer.add(buffer_state, transitions)

        self.env_steps_per_epoch = alg_cfg.num_envs * alg_cfg.iterations_per_epoch
        self.state = SACState(
            nnx.split((network, critic_optimizer, actor_optimizer, alpha_optimizer)),
            env_state,
            buffer_state,
        )
        self.jitted_step = nnx.scan(nnx.jit(self._step_fn))

    def _step_fn(self, state: SACState, key: Key[Array, ""]):
        env_state, buffer_state = state.env_state, state.buffer_state
        network, critic_optimizer, actor_optimizer, alpha_optimizer = nnx.merge(*state.agent_state)
        rollout_key, train_key = jax.random.split(key)
        metrics = {}

        # Generate data
        env_state, trajectory = trajectory_rollout(network, self.env.step, env_state, 1, rollout_key)
        transitions = trajectory_to_transitions(trajectory, env_state)
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
        buffer_state = self.buffer.add(buffer_state, transitions)

        # Update running normalization statistics of observations
        if isinstance(network.running_norm, RunningNorm):
            network.running_norm.update(transitions.obs)

        # Record trajectory metrics
        num_dones = transitions.dones.sum()
        episode_returns = (transitions.episode_returns * transitions.dones).sum() / num_dones
        episode_lengths = (transitions.episode_lengths * transitions.dones).sum() / num_dones

        metrics["return"] = jnp.where(num_dones > 0, episode_returns, -jnp.inf)
        metrics["lengths"] = jnp.where(num_dones > 0, episode_lengths, -jnp.inf)

        # Train
        for _ in range(self.cfg.grad_steps):
            train_key, sample_key, alpha_key, value_key, policy_key = jax.random.split(train_key, 5)

            # Sample
            batch, _, _ = self.buffer.sample(buffer_state, sample_key, self.cfg.minibatch_size)

            # Losses
            if self.cfg.autotune:
                alpha_loss, _ = alpha_update(
                    network, alpha_optimizer, batch, self.target_entropy, self.cfg, alpha_key
                )
                metrics["alpha_loss"] = alpha_loss
            metrics["log_alpha"] = jnp.array(network.log_alpha.param[...])

            value_loss, _ = value_update(network, critic_optimizer, batch, self.cfg, value_key)
            metrics["value_loss"] = value_loss

            actor_loss, _ = policy_update(network, actor_optimizer, batch, self.cfg, policy_key)
            metrics["actor_loss"] = actor_loss

            # EMA
            ema_update(network.critic, network.critic_target, self.cfg.tau)

        agent_state = nnx.split((network, critic_optimizer, actor_optimizer, alpha_optimizer))
        return SACState(agent_state, env_state, buffer_state), metrics

    def step(self, key: Key[Array, ""]):
        keys = jax.random.split(key, self.cfg.iterations_per_epoch)
        self.state, metrics = self.jitted_step(self.state, keys)
        metrics = {k: finite_mean(v) for k, v in metrics.items()}

        return metrics
