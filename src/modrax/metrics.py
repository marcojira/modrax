"""Training metric aggregation."""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float


def mean_episode_metric(values: Float[Array, "..."], dones: Float[Array, "..."]) -> Array:
    """Compute done-weighted mean of a per-step metric."""
    num_dones = jnp.sum(dones)
    return jnp.sum(values * dones) / jnp.maximum(num_dones, 1)


def compute_training_metrics(trajectory: Any):
    """Compute standard training metrics from a trajectory."""
    metrics = {
        "Rew.": trajectory.rewards.sum(axis=1).mean(),
        "Ep.Ret.": mean_episode_metric(trajectory.episode_returns, trajectory.dones),
        "Ep.Len.": mean_episode_metric(trajectory.episode_lengths, trajectory.dones),
    }

    if isinstance(trajectory.info, dict):
        for key, value in trajectory.info.items():
            metrics[f"info/{key}"] = mean_episode_metric(value, trajectory.dones)

    return metrics


def finite_mean(x: Array) -> Array:
    """Return the mean of entries other than negative infinity."""
    mask = x != -jnp.inf
    return jnp.where(mask, x, 0.0).sum() / jnp.maximum(mask.sum(), 1)
