"""Utility functions."""

import json
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from jaxtyping import Array, Key, PyTree
from rich import print

from modrax.env.base import Env, StateWithMetrics
from modrax.network.base import Network
from modrax.optimizer import Optimizer


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


def compute_training_metrics(trajectory: Any):
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
        "Rew.": mean_traj_reward,
        "Ep.Ret.": mean_ep_return,
        "Ep.Len.": mean_ep_length,
    }


def to_python_float(value: Any, ndigits: int = 4) -> float:
    """Convert value to Python float, handling JAX arrays."""
    if hasattr(value, "mean"):
        return round(float(value.item()), ndigits)
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


def make_trajectory_minibatches(data: PyTree, key: Key[Array, ""], minibatch_size: int):
    batch_size = jax.tree_util.tree_leaves(data)[0].shape[0]

    # Shuffle trajectories (permute envs only)
    permutation = jax.random.permutation(key, batch_size)
    data = jax.tree_util.tree_map(lambda x: x[permutation], data)

    num_batches = batch_size // minibatch_size

    # Reshape into minibatches: (B, T, ...) -> (num_batches, minibatch_size, T, ...)
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape(num_batches, minibatch_size, *x.shape[1:]), data
    )

    return minibatches


def make_transition_minibatches(all_data: dict, key: Key[Array, ""], minibatch_size: int):
    """Create minibatches by flattening B x T, shuffling all transitions, then batching."""
    flat_data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), all_data)
    num_samples = jax.tree.leaves(flat_data)[0].shape[0]

    num_batches = num_samples // minibatch_size

    permutation = jax.random.permutation(key, num_samples)
    shuffled_data = jax.tree.map(
        lambda x: x[permutation][: num_batches * minibatch_size], flat_data
    )
    minibatches = jax.tree.map(
        lambda x: x.reshape(num_batches, minibatch_size, *x.shape[1:]), shuffled_data
    )  # Extra dimension to match train_forward requirements
    return minibatches


def update_network_minibatches(
    network: Network, optimizer: Optimizer, minibatches: Any, loss_fn: Callable, config
):
    def _update(graph_state, minibatch: Any):
        graphdef, state = graph_state
        network, optimizer = nnx.merge(graphdef, state)

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(network, minibatch, config)
        optimizer.update(grads)

        graph_state = nnx.split((network, optimizer))
        return graph_state, (loss, info)

    # Scan update over minibatches
    graph_state = nnx.split((network, optimizer))
    graph_state, (loss, infos) = jax.lax.scan(
        _update,
        graph_state,
        minibatches,
    )

    # Update objects after training
    nnx.update((network, optimizer), graph_state[-1])
    return loss, infos


def update_network(network: Network, optimizer: Optimizer, data: Any, loss_fn: Callable, config):
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(network, data, config)
    optimizer.update(grads)
    return loss, info


def finite_mean(x: Array) -> Array:
    """Return the mean of non -inf entries."""
    mask = x != -jnp.inf
    return jnp.where(mask, x, 0.0).sum() / jnp.maximum(mask.sum(), 1)


def ema_update(source: nnx.Module, target: nnx.Module, tau: float):
    """Update target network parameters with exponential moving average of source."""
    source_params = nnx.state(source, nnx.Param)
    target_params = nnx.state(target, nnx.Param)
    new_target_params = jax.tree.map(
        lambda s, t: tau * s + (1 - tau) * t, source_params, target_params
    )
    nnx.update(target, new_target_params)

    # Copy non-parameter state (e.g. BatchNorm running stats) directly
    source_batch_stats = nnx.state(source, nnx.BatchStat)
    nnx.update(target, source_batch_stats)


def render_trajectories(
    env: Env,
    trajectories: StateWithMetrics,  # [T, B, H, W, 3]
) -> list[np.ndarray]:
    """Render trajectories into a list of numpy arrays, one per trajectory.

    Each array has shape (T, H, W, 3) with dtype uint8.
    """
    return [
        env.batch_render(jax.tree.map(lambda x: x[:, i], trajectories))
        for i in range(len(trajectories))
    ]
