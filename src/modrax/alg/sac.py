import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.base import Alg, AlgConfig
from modrax.buffer import ReplayBuffer
from modrax.env.base import Env
from modrax.network.base import Network, NetworkConfig
from modrax.network.block.running_norm import RunningNorm, RunningNormConfig
from modrax.network.module.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.rollout.transitions_rollout import jit_rollout, rollout
from modrax.types import Shape
from modrax.utils import ema_update, finite_mean


class SACNetworkConfig(NetworkConfig):
    model_config = {"frozen": True}

    q_optimizer_config: OptimizerConfig = OptimizerConfig(learning_rate=1e-3)
    actor_optimizer_config: OptimizerConfig = OptimizerConfig(learning_rate=1e-3)
    alpha_optimizer_config: OptimizerConfig = OptimizerConfig(learning_rate=3e-4)

    running_norm: bool = True

    min_std: float = 0.001
    alpha: float = 1.0


class SACConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    minibatch_size: int = 512

    num_envs: int = 128
    learning_start: int = int(5e3)
    autotune: bool = True
    grad_steps: int = 8

    network_config: SACNetworkConfig = SACNetworkConfig()


class Actor(Network):
    def __init__(
        self, obs_shape: Shape, action_size: int, config: SACNetworkConfig, rngs: nnx.Rngs
    ):
        self.fc1 = nnx.Linear(math.prod(obs_shape), 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.fc_mean = nnx.Linear(256, action_size, rngs=rngs)
        self.fc_std = nnx.Linear(256, action_size, rngs=rngs)
        self.min_std = config.min_std

    def __call__(self, x):
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = jax.nn.softplus(self.fc_std(x)) + self.min_std
        return mean, std

    def get_action(self, x: Float[Array, "B D"], key: Key[Array, ""]):
        mean, std = self(x)

        # Reparameterization trick (pre-tanh)
        x_t = mean + std * jax.random.normal(key, mean.shape)

        # Log prob with numerically stable tanh Jacobian correction
        log_prob = -0.5 * ((x_t - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std)
        log_det_jacobian = 2.0 * (jnp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t))
        log_prob = log_prob - log_det_jacobian
        log_prob = jnp.sum(log_prob, axis=-1)

        action = jnp.tanh(x_t)
        return action, log_prob


class Critic(Network):
    def __init__(
        self, obs_shape: Shape, action_size: int, config: SACNetworkConfig, rngs: nnx.Rngs
    ):
        flat_input_size = math.prod(obs_shape) + action_size

        self.soft_q_1 = MLP(
            flat_input_size, (256, 256), 1, activation_fn=jax.nn.relu, rngs=rngs, layer_norm=True
        )
        self.soft_q_2 = MLP(
            flat_input_size, (256, 256), 1, activation_fn=jax.nn.relu, rngs=rngs, layer_norm=True
        )

    def get_qs(self, obs, action):
        x = jnp.concat([obs, action], axis=-1)
        return self.soft_q_1(x).squeeze(-1), self.soft_q_2(x).squeeze(-1)


class LogAlpha(Network):
    def __init__(self, value):
        self.value = nnx.Param(value * jnp.ones((1,)))  # Somehow prevents recompilation?


class SACNetwork(Network):
    def __init__(
        self, obs_shape: Shape, action_size: int, config: SACNetworkConfig, rngs: nnx.Rngs
    ):
        self.critic = Critic(obs_shape, action_size, config, rngs)
        self.critic_target = nnx.clone(self.critic)
        self.actor = Actor(obs_shape, action_size, config, rngs)
        self.log_alpha = LogAlpha(jnp.log(jnp.array(config.alpha)))

        self.running_norm = (
            RunningNorm(obs_shape, math.prod(obs_shape), RunningNormConfig(), rngs=rngs)
            if config.running_norm
            else lambda x: x
        )

        self.critic_optimizer = Optimizer(config.q_optimizer_config, self.critic)
        self.actor_optimizer = Optimizer(config.actor_optimizer_config, self.actor)
        self.alpha_optimizer = Optimizer(config.alpha_optimizer_config, self.log_alpha)

    def get_action(self, obs, key):
        obs = self.running_norm(obs)
        return self.actor.get_action(obs, key)


""" ALGORITHM """


def value_update(network, minibatch, config, key):
    def value_loss(critic, batch, key):
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
        min_q_next_target = min_q - jnp.exp(network.log_alpha.value) * next_action_log_prob
        next_q_value = rewards + (1 - dones) * config.gamma * min_q_next_target

        q1, q2 = critic.get_qs(obs, actions)

        value_loss = jnp.square(q1 - next_q_value) + jnp.square(q2 - next_q_value)
        value_loss = value_loss * (1 - truncations)  # Zero out TD error for truncated transitions
        value_loss = 0.5 * jnp.mean(value_loss)
        return value_loss, {}

    network.critic.train()
    (loss, info), grads = nnx.value_and_grad(value_loss, has_aux=True)(
        network.critic, minibatch, key
    )
    network.critic_optimizer.update(grads)
    network.critic.eval()

    return loss, info


def policy_update(network, minibatch, config, key):
    def policy_loss(actor, batch, key):
        obs = network.running_norm(batch.obs)
        sample_act, log_pi = actor.get_action(obs, key)
        qf1_act, qf2_act = network.critic.get_qs(obs, sample_act)
        min_qf_act = jnp.minimum(qf1_act, qf2_act)
        actor_loss = ((jnp.exp(network.log_alpha.value) * log_pi) - min_qf_act).mean()
        return actor_loss, {}

    network.actor.train()
    (loss, info), grads = nnx.value_and_grad(policy_loss, has_aux=True)(
        network.actor, minibatch, key
    )
    network.actor_optimizer.update(grads)
    network.actor.eval()

    return loss, info


def alpha_update(network, minibatch, target_entropy, config, key):
    def alpha_loss(log_alpha, batch, key):
        obs = network.running_norm(batch.obs)
        _, log_pi = network.actor.get_action(obs, key)
        log_pi = jax.lax.stop_gradient(log_pi)
        alpha_loss = (-jnp.exp(log_alpha.value) * (log_pi + target_entropy)).mean()
        return alpha_loss, {}

    (loss, info), grads = nnx.value_and_grad(alpha_loss, has_aux=True)(
        network.log_alpha, minibatch, key
    )
    network.alpha_optimizer.update(grads)

    return loss, info


class SACAlg(Alg):
    def __init__(self, env: Env, alg_config: SACConfig, key, jit: bool = False):
        self.env = env
        self.config = alg_config
        self.jit = jit

        self.network = SACNetwork(
            env.obs_shape, env.action_size, self.config.network_config, nnx.Rngs(key)
        )
        self.target_entropy = -0.5 * env.action_size
        self.rollout_fn = jit_rollout if jit else rollout

        # Generate transitions for initial buffer
        self.env_state = self.env.reset(jax.random.split(key, self.config.num_envs))
        num_steps = int(self.config.learning_start / self.config.num_envs)

        self.env_state, trajectories = self.rollout_fn(
            self.network, self.env.step, self.env_state, num_steps, key
        )
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), trajectories)

        # Init buffer
        self.buffer = ReplayBuffer(max_size=alg_config.buffer_size, jit=jit)
        self.buffer_state = self.buffer.init(transitions)
        self.buffer_state = self.buffer.add(self.buffer_state, transitions)

        # Iteration function
        self.loop = self.make_loop_fn()

    def make_loop_fn(self):
        def loop(state, key):
            network_graph_state, env_state, buffer_state = state
            network = nnx.merge(*network_graph_state)
            rollout_key, train_key = jax.random.split(key)

            metrics = {}

            # Generate data
            env_state, trajectories = self.rollout_fn(
                network, self.env.step, env_state, 1, rollout_key
            )
            transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), trajectories)
            buffer_state = self.buffer.add(buffer_state, transitions)

            # Update running normalization statistics of observations
            if self.config.network_config.running_norm:
                network.running_norm.update(transitions.obs)

            # Record trajectory metrics
            num_dones = transitions.dones.sum()
            episode_returns = (transitions.episode_returns * transitions.dones).sum() / num_dones
            episode_lengths = (transitions.episode_lengths * transitions.dones).sum() / num_dones

            metrics["return"] = jnp.where(num_dones > 0, episode_returns, -jnp.inf)
            metrics["lengths"] = jnp.where(num_dones > 0, episode_lengths, -jnp.inf)

            # Train
            for _ in range(self.config.grad_steps):
                train_key, sample_key, alpha_key, value_key, policy_key = jax.random.split(
                    train_key, 5
                )

                # Sample
                batch, _, _ = self.buffer.sample(
                    buffer_state, sample_key, self.config.minibatch_size
                )

                # Losses
                if self.config.autotune:
                    alpha_loss, _ = alpha_update(
                        network, batch, self.target_entropy, self.config, alpha_key
                    )
                    metrics["alpha_loss"] = alpha_loss
                metrics["log_alpha"] = jnp.array(network.log_alpha.value)

                value_loss, _ = value_update(network, batch, self.config, value_key)
                metrics["value_loss"] = value_loss

                actor_loss, _ = policy_update(network, batch, self.config, policy_key)
                metrics["actor_loss"] = actor_loss

                # EMA
                ema_update(network.critic, network.critic_target, self.config.tau)

            return (nnx.split(network), env_state, buffer_state), metrics

        loop = nnx.jit(loop) if self.jit else loop
        return nnx.scan(loop)

    def __call__(self, key: Key[Array, ""]):
        keys = jax.random.split(key, self.config.num_gen_steps)

        self.network.eval()
        state = (nnx.split(self.network), self.env_state, self.buffer_state)
        (self.network, self.env_state, self.buffer_state), metrics = self.loop(state, keys)
        self.network = nnx.merge(*self.network)

        metrics = {k: finite_mean(v) for k, v in metrics.items()}
        return metrics
