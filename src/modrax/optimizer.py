from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from pydantic import BaseModel

from modrax.network.base import Network


class OptimizerConfig(BaseModel):
    optimizer_type: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 3e-4
    gradient_clip: float | None = None


class Optimizer(nnx.Optimizer):
    """Optimizer with config-based initialization for consistent interface."""

    def __init__(self, config: OptimizerConfig, network: Network):
        # Create optax optimizer based on config
        if config.optimizer_type == "adam":
            optax_optimizer = optax.adam(config.learning_rate)
        elif config.optimizer_type == "sgd":
            optax_optimizer = optax.sgd(config.learning_rate)
        elif config.optimizer_type == "rmsprop":
            optax_optimizer = optax.rmsprop(config.learning_rate)
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

        # Does a first update using the optimizer (seems to initialize the optimizer state?)
        # This prevents many functions from needing to be compiled twice
        # Only use parameters (nnx.Param), not RNG state or other non-trainable state
        _, params, _ = nnx.split(network, nnx.Param, ...)
        self.update(jax.tree.map(jnp.zeros_like, params))
