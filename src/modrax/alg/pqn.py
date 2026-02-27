from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from jaxtyping import Array, Float, Int, Key

from modrax.alg.base import Alg
from modrax.env.base import Env, StateWithMetrics
from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.rollout.eval_rollout import eval_rollout
from modrax.rollout.trajectory_rollout import Trajectory, trajectory_rollout
from modrax.types import Config
from modrax.utils import (
    compute_training_metrics,
    make_trajectory_minibatches,
    make_transition_minibatches,
    update_network_minibatches,
)


@dataclass
class PQNConfig(Config):
    gamma: float = 0.99
    lambd: float = 0.65

    total_steps: int = int(1e7)
    num_envs: int = 128
    num_timesteps: int = 32
    num_minibatches: int = 32
    num_updates: int = 2

    start_eps: float = 1.0
    end_eps: float = 0.05
    eps_decay: float = 0.1


""" NETWORK """


@struct.dataclass
class PQNNetworkOutput:
    q_values: Any
    carry: Any


class PQNNetwork(Network):
    """Abstract base class for PQN networks."""

    def train_forward(
        self,
        obs: Float[Array, "B T ..."],
        dones: Float[Array, "B T"],
        init_carry,
        saved_carry,
    ):
        raise NotImplementedError

    def policy(
        self, env_state: StateWithMetrics, key: Key[Array, ""]
    ) -> tuple[Int[Array, " B"], PQNNetworkOutput]:
        raise NotImplementedError

    def get_last_q(self, env_state: StateWithMetrics) -> Float[Array, "B A"]:
        raise NotImplementedError


""" HELPERS """


def compute_total_updates(cfg: PQNConfig) -> int:
    num_epochs = cfg.total_steps // (cfg.num_envs * cfg.num_timesteps)
    return num_epochs * cfg.num_updates * cfg.num_minibatches


def compute_targets(
    trajectory: Trajectory,
    last_return: Float[Array, " B"],
    last_q: Float[Array, " B"],
    config: PQNConfig,
) -> Float[Array, "T B"]:
    def backwards_step(lambda_returns_and_next_q, trajectory):
        lambda_returns, next_q = lambda_returns_and_next_q
        done, reward, q_val = (
            trajectory.dones,
            trajectory.rewards,
            trajectory.network_output.q_values,
        )

        target_bootstrap = reward + config.gamma * (1 - done) * next_q
        delta = lambda_returns - next_q
        lambda_returns = target_bootstrap + config.gamma * config.lambd * delta
        lambda_returns = (1 - done) * lambda_returns + done * reward
        next_q = jnp.max(q_val, axis=-1)
        return (lambda_returns, next_q), lambda_returns

    scan_fn = nnx.scan(
        backwards_step, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1), reverse=True
    )
    _, targets = scan_fn((last_return, last_q), trajectory)

    return targets


def pqn_loss(network: PQNNetwork, minibatch, config: PQNConfig):
    batch = minibatch["trajectory"]
    q_values = network.train_forward(
        batch.obs, batch.dones, minibatch["init_carry"], batch.network_output.carry
    )

    chosen_q_values = jnp.take_along_axis(q_values, batch.actions[..., None], axis=-1)
    chosen_q_values = chosen_q_values.squeeze(axis=-1)
    loss = 0.5 * jnp.square(chosen_q_values - minibatch["targets"]).mean()

    return loss, {"loss": loss}


""" ALGORITHM """


@struct.dataclass
class PQNState:
    agent_state: Any
    env_state: StateWithMetrics
    step: int


class PQNAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: Network,
        optimizer: Optimizer,
        cfg: PQNConfig,
        key: Key[Array, ""],
        jit: bool = False,
    ):
        super().__init__(env, network, optimizer, cfg, key, jit)
        self.total_steps = cfg.total_steps
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_timesteps
        self.num_epochs = self.total_steps // self.env_steps_per_epoch
        self.total_updates = compute_total_updates(cfg)

        self.eps_scheduler = optax.linear_schedule(
            cfg.start_eps, cfg.end_eps, int(cfg.eps_decay * self.num_epochs)
        )

        env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        self.state = PQNState(nnx.split((self.network, self.optimizer)), env_state, 0)
        self.loop = nnx.jit(self._loop) if self.jit else self._loop
        self.eval_loop = nnx.jit(self._eval_loop) if self.jit else self._eval_loop

    def _loop(self, state: PQNState, key):
        rollout_key, update_key = jax.random.split(key)
        network, optimizer = nnx.merge(*state.agent_state)

        # Update epsilon
        network.eps = self.eps_scheduler(state.step)

        # Generate data
        init_carry = network.get_carry()
        env_state, trajectory = trajectory_rollout(
            network, self.env.step, state.env_state, self.cfg.num_timesteps, rollout_key
        )

        # Run multiple epochs of updates
        epoch_keys = jax.random.split(update_key, self.cfg.num_updates)
        for epoch_key in epoch_keys:
            # Compute targets
            last_q = jnp.max(network.get_last_q(env_state), axis=-1)
            last_q = last_q * (1 - trajectory.dones[:, -1])
            last_return = trajectory.rewards[:, -1] + self.cfg.gamma * last_q
            targets = compute_targets(trajectory, last_return, last_q, self.cfg)

            all_data = {
                "trajectory": trajectory,
                "targets": targets,
                "init_carry": init_carry,
            }

            if network.is_recurrent:
                minibatch_size = self.cfg.num_envs // self.cfg.num_minibatches
                minibatches = make_trajectory_minibatches(all_data, epoch_key, minibatch_size)
            else:
                minibatch_size = (
                    self.cfg.num_envs * self.cfg.num_timesteps
                ) // self.cfg.num_minibatches
                minibatches = make_transition_minibatches(all_data, epoch_key, minibatch_size)

            loss, infos = update_network_minibatches(
                network, optimizer, minibatches, pqn_loss, self.cfg
            )

        metrics = compute_training_metrics(trajectory)
        metrics.update(jax.tree.map(lambda x: x.mean(), infos))

        return PQNState(nnx.split((network, optimizer)), env_state, state.step + 1), metrics

    def __call__(self, key):
        self.state, metrics = self.loop(self.state, key)
        return metrics

    def _eval_loop(self, network, key):
        eval_env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        episode_returns, episode_lengths, trajectories = eval_rollout(
            network, self.env.step, eval_env_state, key, max_steps=2000
        )
        return {
            "eval_return": episode_returns.mean(),
            "eval_length": episode_lengths.mean(),
        }, trajectories

    def eval(self, key):
        network, _ = nnx.merge(*self.state.agent_state)
        network = nnx.clone(network)
        network.eps = 0.0

        if network.is_recurrent:
            network.reset(jnp.ones(self.cfg.num_envs))

        return self.eval_loop(network, key)
