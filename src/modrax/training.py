"""Training utilities for RL algorithms."""

import dataclasses
import os
import time
from dataclasses import dataclass, field

import imageio.v3 as iio
import wandb

from modrax.alg.base import Alg
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.types import Config

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
from flax import nnx
from rich import print
from tqdm import tqdm

from modrax.env import Env, EnvConfig
from modrax.env.base import StateWithMetrics
from modrax.network.base import Network
from modrax.utils import format_metrics, pprint, render_trajectories, save_metrics_jsonl


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = False
    project: str = ""
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str] | None = None


@dataclass(frozen=True)
class TrainConfig(Config):
    env_cfg: EnvConfig
    network_cfg: Config
    optimizer_cfg: OptimizerConfig
    alg_cfg: Config

    seed: int

    eval_interval: int = 0

    wandb: WandbConfig = field(default_factory=WandbConfig)
    display_network: bool = False
    save_path: str | None = None
    save_gif_wandb: bool = False
    save_gif_local: bool = False
    num_gif_trajectories: int = 5


def log_metrics(metrics: dict, config: TrainConfig, filename: str):
    """Log metrics to wandb and/or save to JSONL."""
    if config.wandb.enabled:
        wandb.log(metrics)
    if config.save_path is not None:
        save_metrics_jsonl(metrics, os.path.join(config.save_path, filename))


def log_trajectories(
    env: Env,
    trajectories: StateWithMetrics,
    config: TrainConfig,
    epoch: int,
    n_trajectories: int = 2,
    fps: int = 10,
):
    """Render trajectories and log as GIFs to wandb and/or save to disk."""
    if not (config.save_gif_local or config.save_gif_wandb):
        return

    rendered = render_trajectories(env, jax.tree.map(lambda x: x[:, :n_trajectories], trajectories))

    for i, frames in enumerate(rendered):
        if config.save_gif_local and config.save_path is not None:
            path = os.path.join(config.save_path, f"trajectory_{epoch}_{i}.gif")
            iio.imwrite(path, frames, extension=".gif", plugin="pillow", loop=0, fps=fps)
        if config.save_gif_wandb and config.wandb.enabled:
            format = "gif" if "minatar" in config.env_cfg.env_name.lower() else "mp4"
            video = wandb.Video(frames.transpose(0, 3, 1, 2), fps=fps, format="format")
            wandb.log({f"eval_trajectories/traj_{i}": video})


def flatten_cfg(config: TrainConfig) -> dict:
    """Flatten TrainConfig into a flat dict with prefixed keys for wandb."""
    flat = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if dataclasses.is_dataclass(value):
            for inner_f in dataclasses.fields(value):
                flat[f"{f.name}.{inner_f.name}"] = getattr(value, inner_f.name)
        else:
            flat[f.name] = value
    return flat


def train(env: Env, network: Network, optimizer: Optimizer, alg: Alg, cfg: TrainConfig) -> Network:
    """Run training loop. Supports both standard and recurrent networks."""
    key = jax.random.key(cfg.seed)

    num_params = sum(p.size for p in jax.tree.leaves(nnx.state(network, nnx.Param)))
    print(f"Training a network with {num_params:} parameters...")

    if cfg.display_network:
        nnx.display(network)

    # Wandb
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            group=cfg.wandb.group,
            tags=cfg.wandb.tags,
            config=flatten_cfg(cfg),
        )

    # Training loop
    num_epochs = alg.total_steps // (alg.env_steps_per_epoch)
    pbar = tqdm(range(num_epochs), desc="Training")
    start = time.time()

    for epoch in pbar:
        key, epoch_key = jax.random.split(key)

        # Alg epoch
        metrics = alg(epoch_key)

        # Metrics
        total_steps = epoch * alg.env_steps_per_epoch
        metrics["epoch"] = epoch
        metrics["steps_M"] = total_steps / 1e6
        metrics["steps/s"] = total_steps / (time.time() - start)

        formatted_metrics = format_metrics(metrics)
        pbar.set_postfix({k: v for k, v in formatted_metrics.items() if not k.startswith("info/")})

        log_metrics(formatted_metrics, cfg, "metrics.jsonl")

        # Evaluation
        if cfg.eval_interval and (epoch % cfg.eval_interval == 0 or epoch == num_epochs - 1):
            eval_key, key = jax.random.split(key)
            eval_metrics, trajectories = alg.eval(eval_key)
            eval_metrics["epoch"] = epoch
            eval_metrics["steps_M"] = total_steps / 1e6
            eval_metrics = {f"eval/{k}": v for k, v in format_metrics(eval_metrics).items()}

            pprint(eval_metrics)
            log_metrics(eval_metrics, cfg, "eval.jsonl")
            log_trajectories(env, trajectories, cfg, epoch, n_trajectories=cfg.num_gif_trajectories)

    # Save checkpoint
    if cfg.save_path is not None:
        checkpoint_path = os.path.join(cfg.save_path, "checkpoint")
        network.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    if cfg.wandb.enabled:
        wandb.finish()

    print("\nTraining completed!")
    return network
