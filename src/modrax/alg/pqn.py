from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax
from flax import nnx, struct
from jaxtyping import Array, Float, Int, Key

from modrax.alg.base import Alg
from modrax.env.base import DiscreteActionSpec, Env, StateWithMetrics
from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.rollout.eval_rollout import eval_rollout
from modrax.rollout.trajectory_rollout import trajectory_rollout
from modrax.types import Config
from modrax.utils import (
    compute_training_metrics,
    make_trajectory_minibatches,
    make_transition_minibatches,
    update_network_minibatches,
)


@dataclass(frozen=True)
class PQNConfig(Config):
    name: str = "PQN"

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

    operator_fn: Callable = lambda q, cfg: jnp.max(q, axis=-1)
    omega: float = 0.5


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


""" HELPERS """


def compute_total_updates(cfg: PQNConfig) -> int:
    num_epochs = cfg.total_steps // (cfg.num_envs * cfg.num_timesteps)
    return num_epochs * cfg.num_updates * cfg.num_minibatches


def compute_targets(
    rewards: Float[Array, "B T"],
    dones: Float[Array, "B T"],
    q_targets: Float[Array, "B T A"],
    config: PQNConfig,
) -> Float[Array, "B T"]:
    def backwards_step(lambda_returns_and_next_q, step_data):
        reward, done, q_val = step_data
        lambda_returns, next_q = lambda_returns_and_next_q

        target_bootstrap = reward + config.gamma * (1 - done) * next_q
        delta = lambda_returns - next_q
        lambda_returns = target_bootstrap + config.gamma * config.lambd * delta
        lambda_returns = (1 - done) * lambda_returns + done * reward
        next_q = config.operator_fn(q_val, config)
        return (lambda_returns, next_q), lambda_returns

    last_q = config.operator_fn(q_targets[:, -1], config)
    lambda_return = rewards[:, -2] + config.gamma * (1 - dones[:, -2]) * last_q
    prev_q = config.operator_fn(q_targets[:, -2], config)

    scan_fn = nnx.scan(
        backwards_step, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1), reverse=True
    )
    _, targets = scan_fn(
        (lambda_return, prev_q),
        (rewards[:, :-2], dones[:, :-2], q_targets[:, :-2]),
    )

    targets = jnp.concatenate([targets, lambda_return[:, None]], axis=1)

    return targets


def pqn_loss(network: PQNNetwork, minibatch, config: PQNConfig):
    batch = minibatch["trajectory"]
    q_values = network.train_forward(batch.obs)

    chosen_q_values = jnp.take_along_axis(q_values, batch.actions[..., None], axis=-1)
    chosen_q_values = chosen_q_values.squeeze(axis=-1)
    loss = 0.5 * jnp.square(chosen_q_values - minibatch["targets"]).mean()

    return loss, {"loss": loss}


def pqn_recurrent_loss(network: PQNNetwork, minibatch, config: PQNConfig):
    traj = minibatch["trajectory"]

    q_values = network.train_forward(
        traj.obs, traj.dones, minibatch["init_carry"], traj.network_output.carry
    )
    target_q_values = jax.lax.stop_gradient(q_values)
    target_q_values = jnp.where(traj.action_masks, target_q_values, -jnp.inf)

    targets = compute_targets(traj.rewards, traj.dones, target_q_values, config)

    chosen_q_values = jnp.take_along_axis(q_values, traj.actions[..., None], axis=-1)
    chosen_q_values = chosen_q_values.squeeze(axis=-1)[:, :-1]
    loss = 0.5 * jnp.square(chosen_q_values - targets).mean()

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
    ):
        if not isinstance(env.action_spec, DiscreteActionSpec):
            raise ValueError("PQN requires a discrete action space")

        super().__init__(env, network, optimizer, cfg, key)
        self.total_steps = cfg.total_steps
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_timesteps
        self.num_epochs = self.total_steps // self.env_steps_per_epoch
        self.total_updates = compute_total_updates(cfg)

        self.eps_scheduler = optax.linear_schedule(
            cfg.start_eps, cfg.end_eps, int(cfg.eps_decay * self.num_epochs)
        )

        env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        self.state = PQNState(nnx.split((self.network, self.optimizer)), env_state, 0)
        self.loop = nnx.jit(self._loop)

    def _loop(self, state: PQNState, key):
        rollout_key, update_key = jax.random.split(key)
        network, optimizer = nnx.merge(*state.agent_state)

        # Update epsilon
        network.eps = nnx.data(self.eps_scheduler(state.step))

        # Generate data
        init_carry = network.get_carry()
        env_state, trajectory = trajectory_rollout(
            network, self.env.step, state.env_state, self.cfg.num_timesteps, rollout_key
        )

        # Run multiple epochs of updates
        epoch_keys = jax.random.split(update_key, self.cfg.num_updates)
        for epoch_key in epoch_keys:
            if network.is_recurrent:
                all_data = {
                    "trajectory": trajectory,
                    "init_carry": init_carry,
                    "env_state": env_state,
                }
                minibatch_size = self.cfg.num_envs // self.cfg.num_minibatches
                minibatches = make_trajectory_minibatches(all_data, epoch_key, minibatch_size)

                loss, infos = update_network_minibatches(
                    network, optimizer, minibatches, pqn_recurrent_loss, self.cfg
                )
            else:
                # If network is not recurrent, targets need to be computed before trajectory gets split into transitions
                q_targets = network.train_forward(
                    trajectory.obs.reshape(-1, *trajectory.obs.shape[2:])
                )
                q_targets = q_targets.reshape(trajectory.obs.shape[0], trajectory.obs.shape[1], -1)
                q_targets = jnp.where(trajectory.action_masks, q_targets, -jnp.inf)

                targets = compute_targets(
                    trajectory.rewards,
                    trajectory.dones,
                    q_targets,
                    self.cfg,
                )
                all_data = {
                    "trajectory": jax.tree.map(lambda x: x[:, :-1], trajectory),
                    "targets": targets,
                }

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

    def eval(self, key):
        network, _ = nnx.merge(*self.state.agent_state)
        network = nnx.clone(network)
        network.eps = 0.0

        if network.is_recurrent:
            network.reset(jnp.ones(self.cfg.num_envs))

        return eval_rollout(self.env, network, self.cfg.num_envs, key, max_steps=1000)
