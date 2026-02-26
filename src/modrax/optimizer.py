from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import optax
from flax import nnx

from modrax.network.base import Network
from modrax.types import Cfg


@dataclass
class OptimizerConfig(Cfg):
    optimizer_type: Literal["adam", "radam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 3e-4
    lr_decay_steps: int | None = None
    gradient_clip: float | None = None


class Optimizer(nnx.Optimizer):
    """Optimizer with config-based initialization for consistent interface."""

    def __init__(self, config: OptimizerConfig, network: Network | nnx.Module):
        # Learning rate (constant or linear decay)
        lr = config.learning_rate
        if config.lr_decay_steps is not None:
            lr = optax.linear_schedule(config.learning_rate, 0.0, config.lr_decay_steps)

        # Create optax optimizer based on config
        if config.optimizer_type == "adam":
            optax_optimizer = optax.adam(lr)
        elif config.optimizer_type == "radam":
            optax_optimizer = optax.radam(lr)
        elif config.optimizer_type == "sgd":
            optax_optimizer = optax.sgd(lr)
        elif config.optimizer_type == "rmsprop":
            optax_optimizer = optax.rmsprop(lr)
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

        # Optionally chain with gradient clipping
        if config.gradient_clip is not None:
            optax_optimizer = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip),
                optax_optimizer,
            )

        # Initialize parent nnx.Optimizer
        super().__init__(network, optax_optimizer)
        self.config = config

        # Warmup update to initialize the optimizer state and prevent recompilation.
        # Uses nnx.value_and_grad to match the real training code path.
        def _warmup(model):
            return jnp.array(0.0), {}

        _, grads = nnx.value_and_grad(_warmup, has_aux=True)(network)
        self.update(grads)
