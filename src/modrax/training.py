"""Training utilities for RL algorithms."""

import os
import time
from dataclasses import dataclass, field

from modrax.alg.base import Alg, AlgConfig
from modrax.logging import (
    WandbConfig,
    finish_logging,
    format_metrics,
    init_logging,
    log_metrics,
    log_trajectories,
    pprint,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
from flax import nnx
from jaxtyping import Array, Key
from rich import print
from tqdm import tqdm

from modrax.env import EnvConfig
from modrax.eval import evaluate
from modrax.network.base import Network, NetworkConfig


@dataclass(frozen=True)
class TrainConfig:
    env_cfg: EnvConfig
    network_cfg: NetworkConfig
    alg_cfg: AlgConfig

    seed: int

    eval_interval: int = 0
    eval_max_steps: int = 1000

    wandb: WandbConfig = field(default_factory=WandbConfig)
    display_network: bool = False
    save_path: str | None = None
    save_gif_wandb: bool = False
    save_gif_local: bool = False
    gif_max_steps: int | None = None
    num_gif_trajectories: int = 5


def train(algorithm: Alg, config: TrainConfig, key: Key[Array, ""]) -> Network:
    """Run training loop. Supports both standard and recurrent networks."""
    env = algorithm.env

    num_params = sum(p.size for p in jax.tree.leaves(nnx.state(algorithm.network, nnx.Param)))
    print(f"Training a network with {num_params:} parameters...")

    if config.display_network:
        nnx.display(algorithm.network)

    init_logging(config)

    # Training loop
    num_epochs = algorithm.cfg.total_steps // algorithm.env_steps_per_epoch
    pbar = tqdm(range(num_epochs), desc="Training")
    start = time.time()

    for epoch in pbar:
        key, epoch_key = jax.random.split(key)

        # Alg epoch
        metrics = algorithm.step(epoch_key)

        # Metrics
        total_steps = epoch * algorithm.env_steps_per_epoch
        metrics["epoch"] = epoch
        metrics["steps_M"] = total_steps / 1e6
        metrics["steps/s"] = total_steps / (time.time() - start)

        formatted_metrics = format_metrics(metrics)
        pbar.set_postfix({k: v for k, v in formatted_metrics.items() if not k.startswith("info/")})

        log_metrics(formatted_metrics, config, "metrics.jsonl")

        # Evaluation
        if config.eval_interval and (
            epoch % config.eval_interval == 0 or epoch == num_epochs - 1
        ):
            eval_key, key = jax.random.split(key)
            eval_metrics, trajectories = evaluate(
                algorithm,
                eval_key,
                max_steps=config.eval_max_steps,
                num_trajectories=config.num_gif_trajectories,
            )
            eval_metrics["epoch"] = epoch
            eval_metrics["steps_M"] = total_steps / 1e6
            eval_metrics = {f"eval/{k}": v for k, v in format_metrics(eval_metrics).items()}

            pprint(eval_metrics)
            log_metrics(eval_metrics, config, "eval.jsonl")
            log_trajectories(
                env,
                trajectories,
                config,
                epoch,
                n_trajectories=config.num_gif_trajectories,
            )
            del eval_metrics, trajectories

    trained_network = algorithm.get_network()

    # Save checkpoint
    if config.save_path is not None:
        checkpoint_path = os.path.join(config.save_path, "checkpoint")
        trained_network.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    finish_logging(config)

    print("\nTraining completed!")
    return trained_network
