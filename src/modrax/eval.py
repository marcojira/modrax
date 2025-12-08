from typing import Any

import jax.numpy as jnp

from modrax.rollout.base import Trajectory


def compute_training_metrics(trajectory: Trajectory, infos: dict[str, Any]) -> dict[str, float]:
    num_dones = jnp.sum(trajectory.dones)

    mean_ep_return = jnp.sum(trajectory.episode_returns * trajectory.dones) / jnp.maximum(
        num_dones, 1
    )
    mean_ep_length = jnp.sum(trajectory.episode_lengths * trajectory.dones) / jnp.maximum(
        num_dones, 1
    )
    mean_traj_reward = trajectory.rewards.sum(axis=0).mean()

    all_metrics = {
        "Loss": infos["total_loss"].mean().item(),
        "Act.": infos["actor_loss"].mean().item(),
        "Crit.": infos["critic_loss"].mean().item(),
        "Ent.": infos["entropy"].mean().item(),
        "Rew.": mean_traj_reward.item(),
        "Ep.Ret.": mean_ep_return.item(),
        "Ep.Len.": mean_ep_length.item(),
    }

    return all_metrics


def format_metrics(metrics: dict[str, float], precision: int = 2) -> dict[str, str]:
    """Format numeric metrics as strings for display."""
    return {key: f"{value:.{precision}f}" for key, value in metrics.items()}
