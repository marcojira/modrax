from typing import Literal

import jax.numpy as jnp
import optax
from flax import nnx
from pydantic import BaseModel

from modrax.network.base import Network


class OptimizerConfig(BaseModel):
    optimizer_type: Literal["adam", "sgd", "rmsprop"] = "adam"
    learning_rate: float = 3e-4
    gradient_clip: float | None = None
    model_config = {"frozen": True}


class Optimizer(nnx.Optimizer):
    """Optimizer with config-based initialization for consistent interface."""

    def __init__(self, config: OptimizerConfig, network: Network | nnx.Module):
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

        # Warmup update to initialize the optimizer state and prevent recompilation.
        # Uses nnx.value_and_grad to match the real training code path.
        def _warmup(model):
            return jnp.array(0.0), {}

        _, grads = nnx.value_and_grad(_warmup, has_aux=True)(network)
        self.update(grads)
