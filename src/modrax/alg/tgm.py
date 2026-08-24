"""TGM, a trajectory-level algorithm for terminal-reward envs (https://github.com/marcojira/tgm)."""

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Bool, Float, Key, Shaped

from modrax.alg.base import Alg, AlgConfig, OptimizerConfig, create_optimizer
from modrax.env.base import DiscreteActionSpec, Env, StateWithMetrics
from modrax.metrics import mean_episode_metric
from modrax.network.base import Network
from modrax.rollout import Trajectory, trajectory_rollout


@dataclass(frozen=True)
class TGMConfig(AlgConfig):
    optimizer_cfg: OptimizerConfig = OptimizerConfig(
        optimizer_type="adamw", learning_rate=1e-4, weight_decay=1e-4, gradient_clip=10.0
    )

    total_steps: int = int(1e7)
    num_envs: int = 16
    num_timesteps: int = 128  # Must be long enough to complete the longest episode

    alpha: float = 1.0
    omega: float = 1.0
    q: float = 1.0
    q_fn: Callable = lambda x: x  # TGM works with the softmax of any function of the logits
    reward_exp: float = 4.0  # Beta in the paper: episode returns enter the loss as exp * log(reward)


""" NETWORK """


class TGMNetwork(Network):
    """Abstract base class for TGM networks."""

    def train_forward(self, obs: Shaped[Array, "B ..."]) -> Float[Array, "B T A"]:
        """Return action logits for a completed episode, position t scoring the step t action."""
        raise NotImplementedError


""" HELPERS """


def masked_log_softmax(
    logits: Float[Array, "... A"], action_mask: Bool[Array, "... A"]
) -> Float[Array, "... A"]:
    return jax.nn.log_softmax(jnp.where(action_mask, logits, -jnp.inf), axis=-1)


def policy_logits(logits: Float[Array, "... A"], cfg: TGMConfig) -> Float[Array, "... A"]:
    """Logits of the policy TGM samples from and optimizes."""
    return cfg.q * cfg.alpha * cfg.q_fn(logits) + cfg.omega * logits


def compute_tgm_metrics(trajectory: Trajectory, reward: Float[Array, " B"]) -> dict:
    """Episode metrics when the reward only arrives once the episode is over."""
    metrics = {
        "Rew.": reward.mean(),
        "Ep.Len.": mean_episode_metric(trajectory.episode_lengths, trajectory.dones),
    }

    for key, value in trajectory.info.items():
        metrics[f"info/{key}"] = mean_episode_metric(value, trajectory.dones)

    return metrics


def tgm_loss(network: TGMNetwork, batch, config: TGMConfig):
    """Variance of the trajectory balance residual across the batch."""
    trajectory = batch["trajectory"]
    num_steps = trajectory.actions.shape[1]

    # Without auto resets the observation stops changing once the episode ends,
    # so the last one holds every action taken
    logits = network.train_forward(trajectory.obs[:, -1])[:, :num_steps]

    log_pi = masked_log_softmax(policy_logits(logits, config), trajectory.action_masks)
    log_ref = masked_log_softmax(config.alpha * config.q_fn(logits), trajectory.action_masks)
    log_ratio = log_pi - config.q * log_ref

    action_log_ratio = jnp.take_along_axis(log_ratio, trajectory.actions[..., None], axis=-1)
    # The terminating action is included: choosing to stop is part of the modelled trajectory
    action_log_ratio = action_log_ratio.squeeze(axis=-1) * trajectory.valid_mask

    residual = batch["log_reward"] - action_log_ratio.sum(axis=1) / config.omega
    loss = residual.var()

    return loss, {"loss": loss, "residual": residual.mean()}


""" ALGORITHM """


@struct.dataclass
class TGMState:
    agent_state: Any


class TGMAlg(Alg):
    cfg: TGMConfig

    def __init__(
        self,
        env: Env,
        network: TGMNetwork,
        cfg: TGMConfig,
        key: Key[Array, ""],
    ):
        if not isinstance(env.action_spec, DiscreteActionSpec):
            raise ValueError("TGM requires a discrete action space")
        if env.config.auto_reset:
            raise ValueError("TGM needs auto_reset=False so that final observations are preserved")
        if not hasattr(env, "terminal_reward"):
            raise ValueError("TGM requires an env that scores finished episodes in terminal_reward")

        super().__init__(env, network, cfg)
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_timesteps

        network.eval()
        optimizer = create_optimizer(network, cfg.optimizer_cfg)

        self.state = TGMState(nnx.split((network, optimizer)))
        self.jitted_step = nnx.jit(self._step_fn)

    def _collect(
        self, network: TGMNetwork, key: Key[Array, ""]
    ) -> tuple[StateWithMetrics, Trajectory]:
        """Generate one complete episode per environment, starting from a fresh reset."""
        reset_key, rollout_key = jax.random.split(key)
        env_state = self.env.reset(jax.random.split(reset_key, self.cfg.num_envs))

        final_env_state, trajectory = trajectory_rollout(
            network,
            self.env.step,
            env_state,
            self.cfg.num_timesteps,
            rollout_key,
            episodic=True,
        )

        return final_env_state, trajectory

    def _step_fn(self, state: TGMState, key: Key[Array, ""]):
        network, optimizer = nnx.merge(*state.agent_state)

        final_env_state, trajectory = self._collect(network, key)
        reward = self.env.terminal_reward(final_env_state)
        all_data = {
            "trajectory": trajectory,
            "log_reward": self.cfg.reward_exp * jnp.log(reward),
        }

        network.train()
        (_, info), grads = nnx.value_and_grad(tgm_loss, has_aux=True)(network, all_data, self.cfg)
        optimizer.update(network, grads)
        network.eval()

        metrics = compute_tgm_metrics(trajectory, reward)
        metrics.update(info)

        return TGMState(nnx.split((network, optimizer))), metrics

    def step(self, key: Key[Array, ""]):
        self.state, metrics = self.jitted_step(self.state, key)
        return metrics
