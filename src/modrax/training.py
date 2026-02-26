"""Training utilities for RL algorithms."""

import os
import time
from dataclasses import dataclass

from modrax.alg.base import Alg
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.types import Cfg

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
from flax import nnx
from rich import print
from tqdm import tqdm

from modrax.env import Env, EnvConfig
from modrax.eval.base import EvalConfig, EvalFn
from modrax.network.base import Network, NetworkConfig
from modrax.utils import format_metrics, pprint, save_metrics_jsonl


@dataclass
class TrainConfig:
    env_config: EnvConfig
    network_config: Cfg
    optimizer_config: OptimizerConfig
    alg_config: Cfg

    seed: int

    eval_interval: int = 0
    jit: bool = True

    display_network: bool = False
    save_path: str | None = None


def train(
    env: Env, network: Network, optimizer: Optimizer, alg: Alg, config: TrainConfig
) -> Network:
    """Run training loop. Supports both standard and recurrent networks."""
    key = jax.random.key(config.seed)

    if config.display_network:
        nnx.display(network)

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
        pbar.set_postfix(formatted_metrics)

        if config.save_path is not None:
            save_metrics_jsonl(formatted_metrics, os.path.join(config.save_path, "metrics.jsonl"))

        # Evaluation
        if config.eval_interval and epoch % config.eval_interval == 0:
            eval_key, key = jax.random.split(key)
            eval_metrics = alg.eval(eval_key)
            eval_metrics = format_metrics(eval_metrics)

            pprint(eval_metrics)
            if config.save_path is not None:
                save_metrics_jsonl(eval_metrics, os.path.join(config.save_path, "eval.jsonl"))

    # Save checkpoint
    if config.save_path is not None:
        checkpoint_path = os.path.join(config.save_path, "checkpoint")
        network.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining completed!")
    return network
