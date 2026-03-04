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
    update_network_minibatches,
)


@dataclass
class PPOConfig(Config):
    name: str = "PPO"

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

    total_steps: int = int(1e7)
    num_envs: int = 1024
    num_gen_steps: int = 128
    num_minibatches: int = 8
    num_updates: int = 3

    start_eps: float = 0.0
    end_eps: float = 0.0
    eps_decay: float = 0.0


""" NETWORK """


@struct.dataclass
class PPONetworkOutput:
    policy: Any
    value: Any
    carry: Any


class PPONetwork(Network):
    """Abstract base class for PPO networks"""

    def train_forward(
        self,
        obs: Float[Array, "B T ..."],
        dones: Float[Array, "B T"],
        init_carry,
        saved_carry,
    ) -> PPONetworkOutput:
        raise NotImplementedError

    def policy(self, env_state, key: Key[Array, ""]) -> tuple[Int[Array, " B"], PPONetworkOutput]:
        raise NotImplementedError


""" HELPERS """


def compute_total_updates(cfg: PPOConfig) -> int:
    num_updates = cfg.total_steps // (cfg.num_envs * cfg.num_gen_steps)
    return num_updates * cfg.num_updates * cfg.num_minibatches


def compute_gae_advantages(
    trajectory: Trajectory, config: PPOConfig
) -> tuple[Float[Array, "B T"], Float[Array, "B T"]]:
    def backwards_fn(gae_and_next_val, transition):
        gae, next_val = gae_and_next_val
        done, val, reward = transition

        delta = reward + config.gamma * next_val * (1 - done) - val
        gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae

        return (gae, val), gae

    values = trajectory.network_output.value.squeeze(-1)  # [B, T]
    last_val = values[:, -1]

    transitions = (trajectory.dones[:, :-1], values[:, :-1], trajectory.rewards[:, :-1])
    _, advantages = nnx.scan(
        backwards_fn, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1), reverse=True
    )((jnp.zeros_like(last_val), last_val), transitions)
    returns = advantages + values[:, :-1]
    return advantages, returns


def ppo_loss(network: PPONetwork, minibatch, config: PPOConfig):
    # Network forward
    batch = minibatch["trajectory"]
    out = network.train_forward(
        batch.obs, batch.dones, minibatch["init_carry"], batch.network_output.carry
    )
    values, action_logits = out.value.squeeze(-1), out.policy
    old_values = batch.network_output.value.squeeze(-1)

    # Apply action mask to logits
    action_mask = batch.action_masks
    masked_logits = jnp.where(action_mask, action_logits, -jnp.inf)
    masked_prev_logits = jnp.where(action_mask, batch.network_output.policy, -jnp.inf)

    prev_log_probs = jax.nn.log_softmax(masked_prev_logits, axis=-1)  # type: ignore
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)  # type: ignore

    # Get log probs for taken actions
    actions = batch.actions[..., None]
    current_log_probs = jnp.take_along_axis(log_probs, actions, axis=-1).squeeze(-1)
    old_log_probs = jnp.take_along_axis(prev_log_probs, actions, axis=-1).squeeze(-1)

    # Compute entropy (only over valid actions)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * jnp.where(action_mask, log_probs, 0.0), axis=-1)

    # Normalize advantages
    advantages = (minibatch["advantage"] - minibatch["advantage"].mean()) / (
        minibatch["advantage"].std() + 1e-8
    )

    # Actor loss with clipping
    prob_ratio = jnp.exp(current_log_probs - old_log_probs)
    clipped_ratio = prob_ratio.clip(1.0 - config.clip_eps, 1.0 + config.clip_eps)
    actor_loss = -jnp.minimum(prob_ratio * advantages, clipped_ratio * advantages)

    # Critic loss with clipping
    value_pred_clipped = old_values + (values - old_values).clip(-config.clip_eps, config.clip_eps)
    value_losses = jnp.square(values - minibatch["return"])
    value_losses_clipped = jnp.square(value_pred_clipped - minibatch["return"])
    critic_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

    # Total loss
    total_loss = (
        actor_loss.mean()
        + config.value_coeff * critic_loss.mean()
        - config.entropy_coeff * entropy.mean()
    )

    info = {
        "actor_loss": actor_loss.mean(),
        "critic_loss": critic_loss.mean(),
        "entropy": entropy.mean(),
        "loss": total_loss,
    }

    return total_loss, info


""" ALGORITHM """


@struct.dataclass
class PPOState:
    agent_state: Any
    env_state: StateWithMetrics
    step: int


class PPOAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: Network,
        optimizer: Optimizer,
        cfg: PPOConfig,
        key: Key[Array, ""],
    ):
        super().__init__(env, network, optimizer, cfg, key)
        self.total_steps = cfg.total_steps
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_gen_steps
        self.num_epochs = self.total_steps // self.env_steps_per_epoch

        self.eps_scheduler = optax.linear_schedule(
            cfg.start_eps, cfg.end_eps, int(cfg.eps_decay * self.num_epochs)
        )

        env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        self.state = PPOState(nnx.split((network, optimizer)), env_state, 0)
        self.loop = nnx.jit(self._loop)

    def _loop(self, state: PPOState, key: Key[Array, ""]):
        rollout_key, update_key = jax.random.split(key)
        network, optimizer = nnx.merge(*state.agent_state)

        # Update epsilon
        network.eps = self.eps_scheduler(state.step)

        # Generate data
        init_carry = network.get_carry()
        env_state, data = trajectory_rollout(
            network, self.env.step, state.env_state, self.cfg.num_gen_steps, rollout_key
        )

        # Bootstrap using last value so [B T] -> [B T-1]
        advantages, returns = compute_gae_advantages(data, self.cfg)

        all_data = {
            "trajectory": jax.tree.map(lambda x: x[:, :-1], data),
            "advantage": advantages,
            "return": returns,
            "init_carry": init_carry,
        }

        # Run multiple epochs of updates
        epoch_keys = jax.random.split(update_key, self.cfg.num_updates)
        for epoch_key in epoch_keys:
            minibatch_size = self.cfg.num_envs // self.cfg.num_minibatches
            minibatches = make_trajectory_minibatches(all_data, epoch_key, minibatch_size)
            loss, infos = update_network_minibatches(
                network, optimizer, minibatches, ppo_loss, self.cfg
            )

        metrics = compute_training_metrics(data)
        metrics.update(jax.tree.map(lambda x: x.mean(), infos))

        return PPOState(nnx.split((network, optimizer)), env_state, state.step + 1), metrics

    def __call__(self, key: Key[Array, ""]):
        self.state, metrics = self.loop(self.state, key)
        return metrics

    def eval(self, key):
        network, _ = nnx.merge(*self.state.agent_state)
        network = nnx.clone(network)
        network.eps = 0.0

        if network.is_recurrent:
            network.reset(jnp.ones(self.cfg.num_envs))

        return eval_rollout(self.env, network, self.cfg.num_envs, key, max_steps=2000)
