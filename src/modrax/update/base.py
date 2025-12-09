from typing import Any, Callable, Protocol

import jax
from flax import nnx
from jaxtyping import Array, Key
from pydantic import BaseModel

from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.rollout.base import RolloutData


class UpdateConfig(BaseModel):
    model_config = {"frozen": True}


class UpdateFn(Protocol):
    def __call__(
        self,
        network: Any,
        optimizer: Optimizer,
        data: RolloutData,
        config: Any,
        key: Key[Array, ""],
    ) -> tuple[Any, dict]: ...


def make_trajectory_minibatches(all_data: dict, key: Key[Array, ""], minibatch_size: int):
    batch_size = all_data["data"].trajectory.obs.shape[0]

    # Shuffle trajectories (permute envs only)
    permutation = jax.random.permutation(key, batch_size)
    all_data = jax.tree_util.tree_map(lambda x: x[permutation], all_data)

    num_batches = batch_size // minibatch_size

    # Reshape into minibatches: (B, T, ...) -> (num_batches, minibatch_size, T, ...)
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape(num_batches, minibatch_size, *x.shape[1:]), all_data
    )

    return minibatches


def update_network(
    network: Network, optimizer: Optimizer, minibatches: Any, loss_fn: Callable, config
):
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
