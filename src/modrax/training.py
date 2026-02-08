"""Training utilities for RL algorithms."""

import os

from modrax.alg.base import Alg, AlgConfig

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import jax
from flax import nnx
from pydantic import BaseModel, SkipValidation
from rich import print
from tqdm import tqdm

from modrax.env import Env, EnvConfig
from modrax.eval import EvalConfig, EvalFn
from modrax.network.base import Network, NetworkConfig
from modrax.network.recurrent_network import RecurrentNetwork
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.utils import format_metrics, pprint, save_metrics_jsonl


class TrainConfig(BaseModel):
    """Configuration for training."""

    model_config = {"arbitrary_types_allowed": True}

    seed: int = 0

    env_config: EnvConfig

    alg_cls: type[Alg]
    alg_config: AlgConfig

    eval_interval: int = 25
    eval_fn: SkipValidation[EvalFn] | None = None
    eval_config: SkipValidation[EvalConfig] | None = None

    num_envs: int
    total_steps: int
    jit: bool = True

    display_network: bool = False
    save_path: str | None = None


def train(config: TrainConfig) -> Network:
    """Run training loop. Supports both standard and recurrent networks."""
    key = jax.random.key(config.seed)

    # Initialize components
    env = Env(config.env_config, jit=config.jit)

    # Initialize state
    reset_key, key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, config.num_envs))

    # recurrent_state = None
    # if isinstance(network, RecurrentNetwork):
    #     recurrent_state = network.init_recurrent_state(config.num_envs)

    key, alg_key = jax.random.split(key)
    alg = config.alg_cls(
        env_state,
        env,  # type: ignore
        config.alg_config,
        alg_key,
        jit=config.jit,
    )

    # if config.display_network:
    #     nnx.display(network)

    # Training loop
    num_iterations = config.total_steps // (config.num_envs * config.alg_config.num_gen_steps)
    pbar = tqdm(range(num_iterations), desc="Training")

    for iteration in pbar:
        key, iteration_key = jax.random.split(key)

        metrics = alg(iteration_key)

        # Metrics
        metrics["iteration"] = iteration
        metrics["steps_M"] = (iteration * config.num_envs * config.alg_config.num_gen_steps) / 1e6
        formatted_metrics = format_metrics(metrics)
        pbar.set_postfix(formatted_metrics)

        if config.save_path is not None:
            save_metrics_jsonl(formatted_metrics, os.path.join(config.save_path, "metrics.jsonl"))

        # Evaluation
        if iteration % config.eval_interval == 0:
            pprint(formatted_metrics)  # Print current metrics

            if config.eval_fn is not None and config.eval_config is not None:
                alg.network.eval()
                eval_key, key = jax.random.split(key)
                eval_metrics = config.eval_fn(alg.network, env, config.eval_config, eval_key)

                pprint(eval_metrics)
                if config.save_path is not None:
                    save_metrics_jsonl(eval_metrics, os.path.join(config.save_path, "eval.jsonl"))

    # Save checkpoint
    if config.save_path is not None:
        checkpoint_path = os.path.join(config.save_path, "checkpoint")
        alg.network.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining completed!")
    return network
