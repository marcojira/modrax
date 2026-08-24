from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Literal

import jax
import optax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.env.base import Env
from modrax.network.base import Network


@dataclass(frozen=True)
class AlgConfig:
    total_steps: int


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_type: Literal["adam", "adamw", "radam", "sgd", "rmsprop", "muon"] = "adam"
    weight_decay: float = 1e-5
    learning_rate: float = 3e-4
    lr_decay: bool = False
    gradient_clip: float | None = None


def create_optimizer(
    network: nnx.Module,
    config: OptimizerConfig,
    total_num_updates: int | None = None,
) -> nnx.Optimizer:
    """Create an NNX optimizer from an algorithm's optimizer configuration."""
    learning_rate = config.learning_rate
    if config.lr_decay:
        assert total_num_updates is not None, "Need total_num_updates when lr_decay is enabled"
        learning_rate = optax.linear_schedule(config.learning_rate, 0.0, total_num_updates)

    if config.optimizer_type == "adam":
        transformation = optax.adam(learning_rate)
    elif config.optimizer_type == "adamw":
        transformation = optax.adamw(
            learning_rate, eps=1e-5, weight_decay=config.weight_decay
        )
    elif config.optimizer_type == "radam":
        transformation = optax.radam(learning_rate)
    elif config.optimizer_type == "sgd":
        transformation = optax.sgd(learning_rate)
    elif config.optimizer_type == "rmsprop":
        transformation = optax.rmsprop(learning_rate)
    elif config.optimizer_type == "muon":
        transformation = optax.contrib.muon(learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

    if config.gradient_clip is not None:
        transformation = optax.chain(
            optax.clip_by_global_norm(config.gradient_clip),
            transformation,
        )

    return nnx.Optimizer(network, transformation, wrt=nnx.Param)


def update_network_minibatches(
    network: nnx.Module,
    optimizer: nnx.Optimizer,
    minibatches: Any,
    loss_fn: Callable[[Network, Any, Any], tuple[Float[Array, ""], dict]],
    config,
):
    """Scan loss computation and gradient updates over minibatches."""

    def _update(graph_state, minibatch: Any):
        graphdef, state = graph_state
        network, optimizer = nnx.merge(graphdef, state)

        (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(network, minibatch, config)
        optimizer.update(network, grads)

        graph_state = nnx.split((network, optimizer))
        return graph_state, (loss, info)

    graph_state = nnx.split((network, optimizer))
    graph_state, (loss, infos) = jax.lax.scan(
        _update,
        graph_state,
        minibatches,
    )

    nnx.update((network, optimizer), graph_state[-1])
    return loss, infos


class Alg(ABC):
    env_steps_per_epoch: int
    state: Any

    def __init__(self, env: Env, network: Network, cfg: AlgConfig):
        self.env = env
        self.network = network
        self.cfg = cfg

    @abstractmethod
    def step(self, key: Key[Array, ""]) -> dict[str, float]:
        """Advance the algorithm by one training iteration."""
        pass

    def get_network(self) -> Network:
        """Return the network from the current algorithm state."""
        return nnx.merge(*self.state.agent_state)[0]
