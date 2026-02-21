from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float, Int, Key

from modrax.alg.base import Alg, AlgConfig
from modrax.env.base import Env, StateWithMetrics
from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.rollout.trajectory_rollout import Trajectory, trajectory_rollout
from modrax.utils import (
    compute_training_metrics,
    make_trajectory_minibatches,
    update_network_minibatches,
)


class PPOConfig(AlgConfig):
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01

    num_envs: int = 1024
    num_gen_steps: int = 128
    minibatch_size: int = 4096
    num_epochs: int = 3


""" NETWORK """


@struct.dataclass
class PPONetworkOutput:
    policy: Any
    value: Any
    carry: Any


class PPONetwork(Network):
    """Abstract base class for PPO networks a"""

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

    def bootstrap_value(self, env_state) -> Float[Array, "B 1"]:
        raise NotImplementedError


""" HELPERS """


def compute_gae_advantages(
    trajectory: Trajectory, last_val: Float[Array, " B"], config: PPOConfig
) -> tuple[Float[Array, "B T"], Float[Array, "B T"]]:
    def backwards_fn(gae_and_next_val, transition):
        gae, next_val = gae_and_next_val
        done, val, reward = transition

        delta = reward + config.gamma * next_val * (1 - done) - val
        gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae

        return (gae, val), gae

    values = trajectory.network_output.value.squeeze(-1)

    # Transpose to (T, B) for scan, then transpose back
    transitions = (trajectory.dones.T, values.T, trajectory.rewards.T)
    _, advantages = jax.lax.scan(
        backwards_fn, (jnp.zeros_like(last_val), last_val), transitions, reverse=True
    )

    advantages = advantages.T
    returns = advantages + values
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
        "total_loss": total_loss,
    }

    return total_loss, info


""" ALGORITHM """


@struct.dataclass
class PPOState:
    agent_state: Any
    env_state: StateWithMetrics


class PPOAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: Network,
        optimizer: Optimizer,
        alg_config: PPOConfig,
        key: Key[Array, ""],
        jit: bool = False,
    ):
        self.env = env
        self.cfg = alg_config
        self.env_steps_per_epoch = alg_config.num_envs * alg_config.num_gen_steps
        self.jit = jit

        env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        self.state = PPOState(nnx.split((network, optimizer)), env_state)
        self.loop = nnx.jit(self._loop) if self.jit else self._loop

    def _loop(self, state: PPOState, key: Key[Array, ""]):
        rollout_key, update_key = jax.random.split(key)
        network, optimizer = nnx.merge(*state.agent_state)

        # Generate data
        init_carry = network.get_carry()
        env_state, data = trajectory_rollout(
            network, self.env.step, state.env_state, self.cfg.num_gen_steps, rollout_key
        )

        # Compute bootstrap value without advancing the carry
        last_val = network.bootstrap_value(env_state).squeeze(-1)
        advantages, returns = compute_gae_advantages(data, last_val, self.cfg)

        all_data = {
            "trajectory": data,
            "advantage": advantages,
            "return": returns,
            "init_carry": init_carry,
        }

        # Run multiple epochs of updates
        epoch_keys = jax.random.split(update_key, self.cfg.num_epochs)
        for epoch_key in epoch_keys:
            minibatches = make_trajectory_minibatches(all_data, epoch_key, self.cfg.minibatch_size)
            loss, infos = update_network_minibatches(
                network, optimizer, minibatches, ppo_loss, self.cfg
            )

        metrics = compute_training_metrics(data)
        metrics.update(jax.tree.map(lambda x: x.mean(), infos))

        return PPOState(nnx.split((network, optimizer)), env_state), metrics

    def __call__(self, key: Key[Array, ""]):
        self.state, metrics = self.loop(self.state, key)
        return metrics
