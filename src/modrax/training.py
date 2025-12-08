"""Training utilities for RL algorithms."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppresses INFO and WARNING messages

import jax
from flax import nnx
from pydantic import BaseModel, SkipValidation
from rich import print
from tqdm import tqdm

from modrax.env import Env, EnvConfig
from modrax.eval import compute_training_metrics, format_metrics
from modrax.network.base import Network, NetworkConfig
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import PolicyFn
from modrax.rollout.base import RolloutConfig, RolloutFn
from modrax.update.base import UpdateConfig, UpdateFn
from modrax.utils import save_metrics_jsonl


class TrainConfig(BaseModel):
    """Base configuration for training."""

    model_config = {"arbitrary_types_allowed": True}

    seed: int = 0

    env_config: EnvConfig
    network_cls: type[Network]
    network_config: NetworkConfig
    optimizer_config: OptimizerConfig

    rollout_fn: SkipValidation[RolloutFn]  # Runtime validaiton of protocols is problematic
    rollout_config: RolloutConfig

    update_fn: SkipValidation[UpdateFn]
    update_config: SkipValidation[UpdateConfig]

    policy_fn: SkipValidation[PolicyFn]

    # Training hyperparameterss
    num_envs: int
    total_steps: int
    num_epochs: int  # Number of updates per generation
    jit: bool = True

    # Logging
    log_interval: int = 25  # Print metrics every N iterations
    save_path: str | None = None  # Path to save metrics as JSONL (None = no saving)


def train(config: TrainConfig) -> Network:
    """Run full training loop."""
    key = jax.random.key(config.seed)
    key, network_key = jax.random.split(key)

    # Initialize components
    env = Env(config.env_config, jit=config.jit)
    network = config.network_cls(
        {"obs": env.obs_shape},
        {"value": 1, "policy": env.num_actions},
        config.network_config,
        nnx.Rngs(network_key),
    )
    optimizer = Optimizer(config.optimizer_config, network)

    rollout_fn = config.rollout_fn
    update_fn = config.update_fn

    if config.jit:
        rollout_fn = nnx.jit(rollout_fn, static_argnames=["policy_fn", "step_fn", "config"])
        update_fn = nnx.jit(update_fn, static_argnames=["config"])

    # Initialize state
    reset_key, key = jax.random.split(key)

    env_state = env.reset(jax.random.split(reset_key, config.num_envs))

    # recurrent_state = None
    # if config.network_config.recurrent_config is not None:
    #     recurrent_state = network.init_recurrent_state(config.num_envs)

    # Training loop
    num_iterations = config.total_steps // (config.num_envs * config.rollout_config.num_steps)
    pbar = tqdm(range(num_iterations), desc="Training")
    for iteration in pbar:
        # Collect trajectory
        rollout_key, key = jax.random.split(key)
        env_state, data = rollout_fn(
            network,
            config.policy_fn,
            env.step,
            env_state,
            config.rollout_config,
            rollout_key,
        )

        # Update for multiple epochs
        for _ in range(config.num_epochs):
            update_key, key = jax.random.split(key)
            loss, infos = update_fn(
                network,
                optimizer,
                data,
                env_state,
                config.update_config,
                update_key,
            )

        # Calculate and log metrics
        metrics = compute_training_metrics(data.trajectory, infos)
        metrics["iteration"] = iteration
        metrics["steps_M"] = (iteration * config.num_envs * config.rollout_config.num_steps) / 1e6

        # Format for display
        formatted_metrics = format_metrics(metrics)
        pbar.set_postfix(formatted_metrics)
        if iteration % config.log_interval == 0:
            print(formatted_metrics)

        # Save raw metrics if path is specified
        if config.save_path is not None:
            save_metrics_jsonl(metrics, os.path.join(config.save_path, "metrics.jsonl"))

    if config.save_path is not None:
        checkpoint_path = os.path.join(config.save_path, "checkpoint")
        network.save(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining completed!")
    return network
