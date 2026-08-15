from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float

from modrax.network.base import Network


@dataclass(frozen=True)
class OptimizerConfig:
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
        self.update(grads)


def update_network_minibatches(
    network: nnx.Module,
    optimizer: Optimizer,
    minibatches: Any,
    loss_fn: Callable[[Network, Any, Any], tuple[Float[Array, ""], dict]],
    config,
):
    """Efficiently scan loss computation and gradient updates over minibatches.
    See https://flax.readthedocs.io/en/stable/guides/performance.html

    loss_fn(network, minibatch, config) -> (scalar_loss, info_dict)
    """

    def _update(graph_state, minibatch: Any):
        graphdef, state = graph_state
        network, optimizer = nnx.merge(graphdef, state)

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(network, minibatch, config)
        optimizer.update(grads)

        graph_state = nnx.split((network, optimizer))
        return graph_state, (loss, info)

    # Scan update over minibatches
    graph_state = nnx.split((network, optimizer))
    graph_state, (loss, infos) = jax.lax.scan(
        _update,
        graph_state,
        minibatches,
    )

    # Update objects after training
    nnx.update((network, optimizer), graph_state[-1])
    return loss, infos


def ema_update(source: nnx.Module, target: nnx.Module, tau: float):
    """Update target network parameters with exponential moving average of source."""
    source_params = nnx.state(source, nnx.Param)
    target_params = nnx.state(target, nnx.Param)
    new_target_params = jax.tree.map(
        lambda s, t: tau * s + (1 - tau) * t, source_params, target_params
    )
    nnx.update(target, new_target_params)

    # Copy non-parameter state (e.g. BatchNorm running stats) directly
    source_batch_stats = nnx.state(source, nnx.BatchStat)
    nnx.update(target, source_batch_stats)
