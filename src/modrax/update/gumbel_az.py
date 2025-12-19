from typing import Callable

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Key

from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.rollout.base import RolloutData, Trajectory
from modrax.update.base import (
    UpdateConfig,
    make_transition_minibatches,
    update_network,
)


class GumbelAZConfig(UpdateConfig):
    model_config = {"frozen": True}

    minibatch_size: int = 32
    sigma_fn: Callable[[Float[Array, "B T A"]], Float[Array, "B T A"]] = lambda x: 5 * x


def compute_value_target(trajectory: Trajectory) -> Float[Array, "B T"]:
    # TODO: This only works for terminal rewards ATM
    def backwards_fn(carry, transition):
        done, reward = transition

        # If done, target is reward, otherwise bring back next value
        value_target = done * reward + (1 - done) * (carry)

        # Inverse value target as carry since previous state is opponent and these are zero-sum
        return -value_target, value_target

    transitions = (trajectory.dones.T, trajectory.rewards.T)

    # Bootstrap with value of network at final step initially
    final_val = trajectory.network_output["value"][:, -1].squeeze(-1)

    _, value_target = jax.lax.scan(backwards_fn, final_val, transitions, reverse=True)
    return value_target.T


def rescale_q_values(q_values: Float[Array, "B T A"], eps: float = 1e-8) -> Float[Array, "B T A"]:
    min_val = jnp.min(q_values, axis=-1, keepdims=True)
    max_val = jnp.max(q_values, axis=-1, keepdims=True)
    return (q_values - min_val) / jnp.maximum(max_val - min_val, eps)


def compute_policy_target(data: RolloutData, config: GumbelAZConfig):
    # TODO Add final target
    traj = data.trajectory
    B, T, num_actions = traj.network_output["policy"].shape

    # Compute Q-value
    # Since this is the value for the opponent, we negate it to get our value (zero-sum game)
    next_val = jnp.roll(traj.network_output["value"].squeeze(-1), shift=-1, axis=-1)
    q_value = jnp.where(traj.dones, traj.rewards, -next_val)  # ??? why negative

    # Create completed Q-values
    v_mix = (traj.network_output["value"].squeeze(-1) + q_value) / 2
    completed_q = jnp.tile(v_mix[..., None], (1, 1, num_actions))
    completed_q = completed_q.at[jnp.arange(B)[:, None], jnp.arange(T)[None, :], traj.actions].set(
        q_value
    )
    completed_q = rescale_q_values(completed_q)

    # Create logits
    logits = traj.network_output["policy"] + config.sigma_fn(completed_q)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(traj.action_masks, logits, -jnp.inf)

    return jax.nn.softmax(logits, axis=-1)


def gumbel_az_loss(network: Network, minibatch, config: GumbelAZConfig):
    out = network.train_forward(minibatch["data"])
    # log_policy = jax.nn.log_softmax(out["policy"], axis=-1)
    # policy_loss = jnp.where(
    #     minibatch["policy_target"] > 1e-8,
    #     minibatch["policy_target"] * (jnp.log(minibatch["policy_target"]) - log_policy),
    #     0,
    # )
    policy_loss = optax.softmax_cross_entropy(out["policy"], minibatch["policy_target"])
    policy_loss = jnp.sum(policy_loss, axis=-1)  # type: ignore
    policy_loss = policy_loss.mean()

    value = out["value"].squeeze(-1)
    value_loss = 0.5 * (value - minibatch["value_target"]) ** 2
    value_loss = value_loss.mean()

    total_loss = policy_loss + value_loss
    metrics = {"policy_loss": policy_loss, "value_loss": value_loss}
    return total_loss, metrics


def gumbel_az_update(
    network: Network,
    optimizer: Optimizer,
    data: RolloutData,
    config: GumbelAZConfig,
    key: Key[Array, ""],
):
    value_target = compute_value_target(data.trajectory)
    policy_target = compute_policy_target(data, config)

    all_data = {"data": data, "value_target": value_target, "policy_target": policy_target}

    minibatches = make_transition_minibatches(all_data, key, config.minibatch_size)

    return update_network(network, optimizer, minibatches, gumbel_az_loss, config)
