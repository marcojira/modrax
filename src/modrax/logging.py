"""Experiment logging utilities."""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v3 as iio
import jax
import numpy as np
import wandb
from rich import print

from modrax.env import Env
from modrax.env.base import StateWithMetrics

if TYPE_CHECKING:
    from modrax.training import TrainConfig


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = ""
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: list[str] | None = None


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


def flatten_config(config: TrainConfig) -> dict[str, Any]:
    """Flatten a training config into prefixed keys for W&B."""
    flat = {}
    for config_field in dataclasses.fields(config):
        value = getattr(config, config_field.name)
        if dataclasses.is_dataclass(value):
            for inner_field in dataclasses.fields(value):
                flat[f"{config_field.name}.{inner_field.name}"] = getattr(
                    value, inner_field.name
                )
        else:
            flat[config_field.name] = value
    return flat


def init_logging(config: TrainConfig) -> None:
    """Initialize enabled experiment loggers."""
    if config.wandb.enabled:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=config.wandb.run_name,
            group=config.wandb.group,
            tags=config.wandb.tags,
            config=flatten_config(config),
        )


def log_metrics(metrics: dict[str, Any], config: TrainConfig, filename: str) -> None:
    """Log metrics to W&B and/or save them as JSONL."""
    if config.wandb.enabled:
        wandb.log(metrics)
    if config.save_path is not None:
        save_metrics_jsonl(metrics, os.path.join(config.save_path, filename))


def save_metrics_jsonl(metrics: dict[str, Any], save_path: str) -> None:
    """Append metrics to a JSONL file."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "a") as file:
        file.write(json.dumps(metrics) + "\n")


def render_trajectories(
    env: Env,
    trajectories: StateWithMetrics,
) -> list[np.ndarray]:
    """Render trajectories into one array per trajectory."""
    num_trajectories = trajectories.obs.shape[0]
    return [
        env.batch_render(jax.tree.map(lambda x: x[i], trajectories))
        for i in range(num_trajectories)
    ]


def log_trajectories(
    env: Env,
    trajectories: StateWithMetrics,
    config: TrainConfig,
    epoch: int,
    n_trajectories: int = 2,
    fps: int = 10,
) -> None:
    """Render trajectories and log them to W&B and/or save them as GIFs."""
    if not (config.save_gif_local or (config.save_gif_wandb and config.wandb.enabled)):
        return

    max_steps = config.gif_max_steps
    rendered = render_trajectories(
        env,
        jax.tree.map(lambda x: x[:n_trajectories, :max_steps], trajectories),
    )

    for index, frames in enumerate(rendered):
        if config.save_gif_local and config.save_path is not None:
            path = os.path.join(config.save_path, f"trajectory_{epoch}_{index}.gif")
            iio.imwrite(path, frames, extension=".gif", plugin="pillow", loop=0, fps=fps)
        if config.save_gif_wandb and config.wandb.enabled:
            video_format = "gif" if "minatar" in config.env_cfg.env_name.lower() else "mp4"
            video = wandb.Video(
                frames.transpose(0, 3, 1, 2), fps=fps, format=video_format
            )
            wandb.log({f"eval_trajectories/traj_{index}": video})


def finish_logging(config: TrainConfig) -> None:
    """Finish enabled experiment loggers."""
    if config.wandb.enabled:
        wandb.finish()
