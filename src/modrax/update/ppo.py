import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key

from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.rollout.base import RolloutData, Trajectory
from modrax.update.base import UpdateConfig, make_trajectory_minibatches, update_network


class PPOConfig(UpdateConfig):
    model_config = {"frozen": True}

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    minibatch_size: int = 4096


""" HELPERS """


def compute_gae_advantages(
    trajectory: Trajectory, last_val: Float[Array, " B"], config: PPOConfig
) -> tuple[Float[Array, "T B"], Float[Array, "T B"]]:
    def backwards_fn(gae_and_next_val, transition):
        gae, next_val = gae_and_next_val
        done, val, reward = transition

        delta = reward + config.gamma * next_val * (1 - done) - val
        gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae

        return (gae, val), gae

    values = trajectory.network_output["value"].squeeze(-1)  # type: ignore
    transitions = (trajectory.dones, values, trajectory.rewards)
    _, advantages = jax.lax.scan(
        backwards_fn,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )

    returns = advantages + values
    return advantages, returns


def ppo_loss(batch: dict, values, action_logits, config: PPOConfig):
    # Apply action mask to logits (Array cast to deal with typing)
    action_mask = batch["action_masks"]
    masked_logits = jnp.asarray(jnp.where(action_mask, action_logits, -jnp.inf))
    masked_prev_logits = jnp.asarray(
        jnp.where(action_mask, batch["network_output"]["policy"], -jnp.inf)
    )

    prev_log_probs = jax.nn.log_softmax(masked_prev_logits, axis=-1)
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)

    # Compute entropy (only over valid actions)
    current_log_probs = jnp.take_along_axis(
        log_probs, batch["actions"][..., None], axis=-1
    ).squeeze(-1)
    old_log_probs = jnp.take_along_axis(
        prev_log_probs, batch["actions"][..., None], axis=-1
    ).squeeze(-1)
    probs = jax.nn.softmax(masked_logits, axis=-1)

    safe_log_probs = jnp.where(action_mask, log_probs, 0.0)
    entropy = -jnp.sum(probs * safe_log_probs, axis=-1)

    advantages = (batch["advantage"] - batch["advantage"].mean()) / (
        batch["advantage"].std() + 1e-8
    )

    # Actor loss with clipping
    prob_ratio = jnp.exp(current_log_probs - old_log_probs)
    clipped_ratio = prob_ratio.clip(1.0 - config.clip_eps, 1.0 + config.clip_eps)
    actor_loss = -jnp.minimum(prob_ratio * advantages, clipped_ratio * advantages)

    # Critic loss with clipping
    value_pred_clipped = batch["network_output"]["value"].squeeze(-1) + (
        values - batch["network_output"]["value"].squeeze(-1)
    ).clip(-config.clip_eps, config.clip_eps)
    value_losses = jnp.square(values - batch["return"])
    value_losses_clipped = jnp.square(value_pred_clipped - batch["return"])
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


""" UPDATE FN"""


def ppo_update(
    network: Network,
    optimizer: Optimizer,
    data: RolloutData,
    final_state,
    config: PPOConfig,
    key: Key[Array, ""],
):
    def loss_fn(network, minibatch, config):
        # Flatten to [B*T, ...]
        minibatch = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), minibatch)

        out = network({"obs": minibatch["obs"]})
        values, action_logits = out["value"], out["policy"]
        values = values.squeeze(-1)
        return ppo_loss(minibatch, values, action_logits, config)

    # Compute last value for GAE bootstrapping
    out = network({"obs": final_state.obs})
    advantages, returns = compute_gae_advantages(data.trajectory, out["value"].squeeze(-1), config)

    # Add advantages and returns to trajectory
    all_data = {
        **data.trajectory._asdict(),
        "advantage": advantages,
        "return": returns,
    }

    minibatches = make_trajectory_minibatches(all_data, key, config.minibatch_size)

    return update_network(network, optimizer, minibatches, loss_fn, config)
