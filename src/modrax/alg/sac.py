import functools
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
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy, uniform_policy
from modrax.rollout import RolloutData, Trajectory
from modrax.rollout.base import RolloutData
from modrax.rollout.transitions_rollout import jit_rollout, rollout
from modrax.types import Shape


class SACNetworkConfig(NetworkConfig):
    model_config = {"frozen": True}
    pass


class SACConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    buffer_size: int = int(1e6)
    gamma: float = 0.99
    tau: float = 0.005
    minibatch_size: int = 512

    num_fill_envs: int = 32
    learning_start: int = int(5e3)
    policy_frequency: int = 2
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = False
    grad_steps: int = 8

    q_optimizer_config: OptimizerConfig = OptimizerConfig(learning_rate=1e-3)
    actor_optimizer_config: OptimizerConfig = OptimizerConfig(learning_rate=3e-4)
    network_config: SACNetworkConfig = SACNetworkConfig()


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(Network):
    def __init__(self, obs_shape: Shape, action_dim: int, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(math.prod(obs_shape), 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.fc_mean = nnx.Linear(256, action_dim, rngs=rngs)
        self.fc_logstd = nnx.Linear(256, action_dim, rngs=rngs)

        self.action_scale = nnx.Param(jnp.ones((action_dim,)))
        self.action_bias = nnx.Param(jnp.zeros((action_dim,)))

    def __call__(self, x):
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        mean = self.fc_mean(x)

        log_std = self.fc_logstd(x)
        log_std = jax.nn.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def get_action(self, x: Float[Array, "B D"], key: Key[Array, ""]):
        mean, log_std = self(x)
        std = jnp.exp(log_std)
        # Reparameterization trick: mean + std * N(0,1)
        x_t = mean + std * jax.random.normal(key, mean.shape)
        y_t = jnp.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        # Log prob with action bound correction
        log_prob = -0.5 * ((x_t - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std)
        log_prob -= jnp.log(self.action_scale * (1 - y_t**2) + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
        mean = jnp.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, x_t

    def get_log_prob(self, x: Float[Array, "B D"], x_t: Float[Array, "B A"]):
        mean, log_std = self(x)
        std = jnp.exp(log_std)
        y_t = jnp.tanh(x_t)
        log_prob = -0.5 * ((x_t - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std)
        log_prob -= jnp.log(self.action_scale * (1 - y_t**2) + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)
        return log_prob


class Critic(Network):
    def __init__(self, obs_shape: Shape, action_dim: int, rngs: nnx.Rngs):
        self.soft_q_1 = MLP(
            math.prod(obs_shape) + action_dim, (256, 256), 1, activation_fn=jax.nn.relu, rngs=rngs
        )
        self.soft_q_2 = MLP(
            math.prod(obs_shape) + action_dim, (256, 256), 1, activation_fn=jax.nn.relu, rngs=rngs
        )

    def get_qs(self, obs, action):
        x = jnp.concat([obs, action], axis=-1)
        return self.soft_q_1(x).squeeze(-1), self.soft_q_2(x).squeeze(-1)


""" ALGORITHM """


def value_loss(critic, critic_target, actor, batch, config, key):
    obs, next_obs, actions, rewards, dones = (
        batch.obs,
        batch.next_obs,
        batch.actions,
        batch.rewards,
        batch.dones,
    )

    next_action, next_action_log_prob, _, _ = actor.get_action(next_obs, key)

    next_action, next_action_log_prob = (
        jax.lax.stop_gradient(next_action),
        jax.lax.stop_gradient(next_action_log_prob),
    )

    q1, q2 = critic_target.get_qs(next_obs, next_action)
    min_q = jnp.minimum(q1, q2)
    min_q_next_target = min_q - config.alpha * next_action_log_prob.squeeze(-1)
    next_q_value = rewards + (1 - dones) * config.gamma * min_q_next_target

    q1, q2 = critic.get_qs(obs, actions)

    value_loss = (
        optax.losses.squared_error(q1, next_q_value).mean()
        + optax.losses.squared_error(q2, next_q_value).mean()
    )
    return value_loss, {}


def value_update(critic, critic_target, actor, optimizer, minibatch, config, key):
    (loss, info), grads = nnx.value_and_grad(value_loss, has_aux=True)(
        critic, critic_target, actor, minibatch, config, key
    )
    optimizer.update(grads)

    return loss, info


def policy_loss(actor, critic, minibatch, config, key):
    sample_act, log_pi, _, _ = actor.get_action(minibatch.obs, key)
    qf1_act, qf2_act = critic.get_qs(minibatch.obs, sample_act)
    min_qf_act = jnp.minimum(qf1_act, qf2_act)
    actor_loss = ((config.alpha * log_pi) - min_qf_act).mean()

    if config.autotune:
        raise NotImplementedError("Autotune not implemented")

    return actor_loss, {}


def policy_update(actor, critic, optimizer, minibatch, config, key):
    (loss, info), grads = nnx.value_and_grad(policy_loss, has_aux=True)(
        actor, critic, minibatch, config, key
    )
    optimizer.update(grads)

    return loss, info


class SACAlg(Alg):
    def __init__(
        self,
        env_state: EnvState,
        env: Env,
        alg_config: SACConfig,
        key,
        jit: bool = False,
    ):
        self.env_state = env_state

        self.env = env
        self.config = alg_config
        self.jit = jit

        self.critic = Critic(
            env.obs_shape,
            env.action_dim,
            nnx.Rngs(key),
        )
        self.actor = Actor(
            env.obs_shape,
            env.action_dim,
            nnx.Rngs(key),
        )
        self.critic_optimizer = Optimizer(alg_config.q_optimizer_config, self.critic)
        self.actor_optimizer = Optimizer(alg_config.actor_optimizer_config, self.actor)

        self.critic_target = nnx.clone(self.critic)

        self.rollout_fn = jit_rollout if jit else rollout

        self.buffer = ReplayBuffer(max_size=alg_config.buffer_size, jit=jit)
        env_state = self.env.reset(jax.random.split(key, self.config.num_fill_envs))
        num_steps = int(self.config.learning_start / self.config.num_fill_envs)

        # Initial buffer fillup
        env_state, trajectories = self.rollout_fn(
            self.actor,
            self.env.step,
            self.env_state,
            num_steps,
            key,
        )
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), trajectories)
        self.buffer_state = self.buffer.init(transitions)
        self.buffer_state = self.buffer.add(self.buffer_state, transitions)

        self.env_state = env_state

        # nnx.display(self.network)
        self.value_update_fn = (
            nnx.jit(value_update, static_argnames=["config"]) if jit else value_update
        )
        self.policy_update_fn = (
            nnx.jit(policy_update, static_argnames=["config"]) if jit else policy_update
        )

        self.jitted_loop = self.make_loop_fn()

        self.iter = 0

    def make_loop_fn(self):
        def loop(state, key):
            (
                actor_graph_state,
                actor_optimizer_graph_state,
                critic_graph_state,
                critic_target_graph_state,
                critic_optimizer_graph_state,
                env_state,
                buffer_state,
                iter,
            ) = state
            rollout_key, update_key = jax.random.split(key)
            metrics = {}
            # Generate data
            actor = nnx.merge(*actor_graph_state)
            actor.eval()
            env_state, trajectories = self.rollout_fn(
                actor,
                self.env.step,
                env_state,
                self.config.num_gen_steps,
                rollout_key,
            )

            transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), trajectories)
            buffer_state = self.buffer.add(buffer_state, transitions)

            # if jnp.any(transitions.dones):
            #     episode_return = transitions.episode_returns.mean().item()
            #     metrics["episode_returns"] = episode_return
            #     print(f"Episode return: {episode_return:.3f}")

            for _ in range(self.config.grad_steps):
                # Train on data
                sample_key, train_key = jax.random.split(update_key)

                actor.train()
                batch, _, _ = self.buffer.sample(
                    buffer_state, sample_key, self.config.minibatch_size
                )

                critic = nnx.merge(*critic_graph_state)
                critic_target = nnx.merge(*critic_target_graph_state)
                critic_optimizer = nnx.merge(*critic_optimizer_graph_state)
                loss, info = self.value_update_fn(
                    critic,
                    critic_target,
                    actor,
                    critic_optimizer,
                    batch,
                    self.config,
                    train_key,
                )
                metrics["value_loss"] = loss

                state = nnx.state(critic)
                target_state = nnx.state(critic_target)
                new_target_state = jax.tree.map(
                    lambda p, tp: self.config.tau * p + (1 - self.config.tau) * tp,
                    state,
                    target_state,
                )
                nnx.update(critic, new_target_state)

                # Policy loss
                actor_optimizer = nnx.merge(*actor_optimizer_graph_state)
                for _ in range(self.config.policy_frequency):
                    loss, info = self.policy_update_fn(
                        actor,
                        critic,
                        actor_optimizer,
                        batch,
                        self.config,
                        train_key,
                    )
                metrics["actor_loss"] = loss

                iter += 1

            return (
                nnx.split(actor),
                nnx.split(actor_optimizer),
                nnx.split(critic),
                nnx.split(critic_target),
                nnx.split(critic_optimizer),
                env_state,
                buffer_state,
                iter,
            ), None

        loop = nnx.jit(loop) if self.jit else loop
        return nnx.scan(loop)

    def __call__(self, key: Key[Array, ""]):
        keys = jax.random.split(key, 100)

        x = self.jitted_loop(
            (
                nnx.split(self.actor),
                nnx.split(self.actor_optimizer),
                nnx.split(self.critic),
                nnx.split(self.critic_target),
                nnx.split(self.critic_optimizer),
                self.env_state,
                self.buffer_state,
                jnp.array(0),
            ),
            keys,
        )
        return {}
        # for _ in range(100):
        #     self.jitted_loop((graph_state, self.env_state), keys[0])
        # state, _ = nnx.scan(self.jitted_loop)((graph_state, self.env_state), keys)
        return {}
