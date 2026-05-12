from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import optax
from flax import nnx

from modrax.network.base import Network
from modrax.types import Config


@dataclass(frozen=True)
class OptimizerConfig(Config):
    optimizer_type: Literal["adam", "adamw", "radam", "sgd", "rmsprop", "muon"] = "adam"
    weight_decay: float = 1e-5
    learning_rate: float = 3e-4
    lr_decay: bool = False
    gradient_clip: float | None = None


class Optimizer(nnx.Optimizer):
    """Optimizer with config-based initialization for consistent interface."""

    def __init__(
        self,
        config: OptimizerConfig,
        network: Network | nnx.Module,
        total_num_updates: int | None = None,
    ):
        # Learning rate (constant or linear decay)
        lr = config.learning_rate
        if config.lr_decay:
            assert total_num_updates is not None, "Need to pass `total_num_updates` to Optimizer"
            lr = optax.linear_schedule(config.learning_rate, 0.0, total_num_updates)

        # Create optax optimizer based on config
        if config.optimizer_type == "adam":
            optax_optimizer = optax.adam(lr)
        elif config.optimizer_type == "adamw":
            optax_optimizer = optax.adamw(lr, weight_decay=config.weight_decay)
        elif config.optimizer_type == "radam":
            optax_optimizer = optax.radam(lr)
        elif config.optimizer_type == "sgd":
            optax_optimizer = optax.sgd(lr)
        elif config.optimizer_type == "rmsprop":
            optax_optimizer = optax.rmsprop(lr)
        elif config.optimizer_type == "muon":
            optax_optimizer = optax.contrib.muon(lr)
        else:
            raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

        # Optionally chain with gradient clipping
        if config.gradient_clip is not None:
            optax_optimizer = optax.chain(
                optax.clip_by_global_norm(config.gradient_clip),
                optax_optimizer,
            )

        # Initialize parent nnx.Optimizer
        super().__init__(network, optax_optimizer, wrt=nnx.Param)
        self.config = config

        # Warmup update to initialize the optimizer state and prevent recompilation.
        # Uses nnx.value_and_grad to match the real training code path.
        def _warmup(model):
            return jnp.array(0.0), {}

        _, grads = nnx.value_and_grad(_warmup, has_aux=True)(network)
        self.update(network, grads)
