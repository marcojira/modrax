from dataclasses import dataclass
from typing import Any, Callable

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
from modrax.rollout.trajectory_rollout import trajectory_rollout
from modrax.types import Config
from modrax.utils import (
    compute_training_metrics,
    make_trajectory_minibatches,
    update_network_minibatches,
)


@dataclass
class TGMConfig(Config):
    name: str = "TGM"

    gamma: float = 0.9999

    total_steps: int = int(1e7)
    num_envs: int = 128
    num_timesteps: int = 32
    num_minibatches: int = 4
    num_updates: int = 2

    start_eps: float = 1.0
    end_eps: float = 0.05
    eps_decay: float = 0.1

    alpha: float = 1.0
    q: float = 1.0
    omega: float = 16.0


""" NETWORK """


@struct.dataclass
class TGMNetworkOutput:
    q_values: Any
    value: Any
    carry: Any


class TGMNetwork(Network):
    """Abstract base class for TGM networks."""

    is_recurrent = False

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
    ) -> tuple[Int[Array, " B"], TGMNetworkOutput]:
        raise NotImplementedError


""" HELPERS """


def compute_total_updates(cfg: TGMConfig) -> int:
    num_epochs = cfg.total_steps // (cfg.num_envs * cfg.num_timesteps)
    return num_epochs * cfg.num_updates * cfg.num_minibatches


def tgm_loss(network: TGMNetwork, minibatch, cfg: TGMConfig):
    traj = minibatch["trajectory"]

    if network.is_recurrent:
        q_values, value = network.train_forward(
            traj.obs, traj.dones, minibatch["init_carry"], traj.network_output.carry
        )
    else:
        q_values, value = network.train_forward(traj.obs)

    lsm_diff = jax.nn.log_softmax(q_values, axis=-1)
    lsm_diff = lsm_diff - cfg.q * jax.nn.log_softmax(
        (cfg.alpha / (cfg.alpha * cfg.q + cfg.omega)) * q_values, axis=-1
    )
    lsm_diff = jnp.take_along_axis(lsm_diff, traj.actions[..., None], axis=-1).squeeze(axis=-1)

    def step(carry, rewards, dones, lsm_diff, value):
        accum, t = carry

        discounts = jnp.pow(cfg.gamma * jnp.ones_like(t), t)
        accum = accum - discounts * rewards + (1 / cfg.omega) * discounts * lsm_diff

        losses = jnp.where(dones, 0.5 * (accum**2), 0)
        accum = jnp.where(dones, value, accum)
        t = jnp.where(dones, 0, t + 1)

        return (accum, t), losses

    B = value.shape[0]
    (accum, t), losses = nnx.scan(step, in_axes=(nnx.Carry, 1, 1, 1, 1), out_axes=(nnx.Carry, 1))(
        (value[:, 0], jnp.zeros(B)),
        traj.rewards[:, :-1],
        traj.dones[:, :-1],
        lsm_diff[:, :-1],
        jax.lax.stop_gradient(value[:, 1:]),  # Bootstrap targets for reset
    )
    loss = jnp.sum(losses)  # type: ignore
    final_discounts = jnp.pow(cfg.gamma * jnp.ones_like(t), t)
    final_losses = 0.5 * ((accum - final_discounts * jax.lax.stop_gradient(value[:, -1])) ** 2)
    loss += jnp.sum(final_losses)
    loss = loss / (jnp.sum(traj.dones) + B)  # Divide by total number of subtrajectories
    return loss, {"loss": loss}


""" ALGORITHM """


@struct.dataclass
class TGMState:
    agent_state: Any
    env_state: StateWithMetrics
    step: int


class TGMAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: Network,
        optimizer: Optimizer,
        cfg: TGMConfig,
        key: Key[Array, ""],
    ):
        super().__init__(env, network, optimizer, cfg, key)
        self.total_steps = cfg.total_steps
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_timesteps
        self.num_epochs = self.total_steps // self.env_steps_per_epoch
        self.total_updates = compute_total_updates(cfg)

        self.eps_scheduler = optax.linear_schedule(
            cfg.start_eps, cfg.end_eps, int(cfg.eps_decay * self.num_epochs)
        )

        env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        self.state = TGMState(nnx.split((self.network, self.optimizer)), env_state, 0)
        self.loop = nnx.jit(self._loop)

    def _loop(self, state: TGMState, key):
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
            all_data = {
                "trajectory": trajectory,
                "init_carry": init_carry,
                "env_state": env_state,
            }
            minibatch_size = self.cfg.num_envs // self.cfg.num_minibatches
            minibatches = make_trajectory_minibatches(all_data, epoch_key, minibatch_size)

            loss, infos = update_network_minibatches(
                network, optimizer, minibatches, tgm_loss, self.cfg
            )

        metrics = compute_training_metrics(trajectory)
        metrics.update(jax.tree.map(lambda x: x.mean(), infos))

        return TGMState(nnx.split((network, optimizer)), env_state, state.step + 1), metrics

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
