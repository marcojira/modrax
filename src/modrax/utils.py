"""Utility functions."""

import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from rich import print


def fig_to_rgb_array(fig: matplotlib.figure.Figure) -> np.ndarray:
    """Convert matplotlib figure to RGB array.

    Args:
        fig: Matplotlib figure

    Returns:
        RGB array of shape (H, W, 3) with dtype uint8
    """
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()  # type: ignore
    width, height = fig.canvas.get_width_height()
    rgb_array = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)[:, :, :3]
    plt.close(fig)
    return rgb_array


def compute_training_metrics(trajectory: Any) -> dict[str, float]:
    """Compute standard training metrics from trajectory and update infos."""
    num_dones = jnp.sum(trajectory.dones)

    mean_ep_return = jnp.sum(trajectory.episode_returns * trajectory.dones) / jnp.maximum(
        num_dones, 1
    )
    mean_ep_length = jnp.sum(trajectory.episode_lengths * trajectory.dones) / jnp.maximum(
        num_dones, 1
    )
    mean_traj_reward = trajectory.rewards.sum(axis=1).mean()

    return {
        "Rew.": mean_traj_reward.item(),
        "Ep.Ret.": mean_ep_return.item(),
        "Ep.Len.": mean_ep_length.item(),
    }


def to_python_float(value: Any, ndigits: int = 4) -> float:
    """Convert value to Python float, handling JAX arrays."""
    if hasattr(value, "mean"):
        return round(float(value.mean().item()), ndigits)
    return round(float(value), ndigits)


def format_metrics(metrics: dict[str, Any], precision: int = 3) -> dict[str, str]:
    """Format numeric metrics as strings for display."""
    formatted = {}
    for key, value in metrics.items():
        value = to_python_float(value, precision)
        formatted[key] = value
    return formatted


def pprint(d: dict[str, Any], ndigits: int = 3) -> None:
    """Pretty print a nested dict with formatted float leaves."""

    def format_value(value: Any):
        if isinstance(value, dict):
            return {k: format_value(v) for k, v in value.items()}
        elif isinstance(value, float) or isinstance(value, int):
            return round(value, ndigits)
        else:
            return str(value)

    print(format_value(d))
    return


def save_metrics_jsonl(metrics: dict[str, Any], save_path: str) -> None:
    """Save metrics to a JSONL file.

    Args:
        metrics: Dictionary of metrics to save
        save_path: Path to the JSONL file (will be created if it doesn't exist)
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # Append metrics as a single JSON line
    with open(save_path, "a") as f:
        f.write(json.dumps(metrics) + "\n")
