"""Utility functions."""

import typing
from typing import Any, Callable

import jax
from flax import nnx
from jaxtyping import Array, Key, PyTree, Shaped

from modrax.network.base import Network
from modrax.optimizer import Optimizer
from modrax.types import Config

Out = typing.TypeVar("Out")
ConfigType = typing.TypeVar("ConfigType", bound=Config)


def add_cli(fn: Callable[[ConfigType], Out]):
    """Adds the simple command-line interface to run this function.

    The wrapped function should accept a dataclass as its first (and only) argument.
    When the wrapped function is called with no arguments, this dataclass is obtained
    from the command-line. When a value is passed, the function behaves as usual.
    """
    import rich_argparse
    import simple_parsing

    # Make the CLI nicer to look at.
    class _FormatterClass(rich_argparse.RichHelpFormatter, simple_parsing.SimpleHelpFormatter): ...  # type: ignore

    # @functools.wraps(fn)
    def wrapper(cfg: ConfigType | None = None) -> Out:
        # If a config is passed, use it. If not, get one from the command-line arguments.
        if not cfg:
            # Inspect the function to figure out the type of config that needs to be parsed.
            config_type = typing.get_type_hints(fn).popitem()[1]
            cfg = simple_parsing.parse(
                config_type, formatter_class=_FormatterClass, description=fn.__doc__
            )
            assert cfg
        return fn(cfg)

    return wrapper


def batch_trajectories(
    data: PyTree[Shaped[Array, "B ..."]], key: Key[Array, ""], minibatch_size: int
):
    """Shuffle and batch a PyTree whose array leaves share a leading environment axis."""
    batch_size = jax.tree_util.tree_leaves(data)[0].shape[0]

    # Shuffle trajectories (permute envs only)
    permutation = jax.random.permutation(key, batch_size)
    data = jax.tree_util.tree_map(lambda x: x[permutation], data)

    num_batches = batch_size // minibatch_size

    # Reshape into minibatches: (B, T, ...) -> (num_batches, minibatch_size, T, ...)
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape(num_batches, minibatch_size, *x.shape[1:]), data
    )

    return minibatches


def batch_transitions(
    data: PyTree[Shaped[Array, "B T ..."]], key: Key[Array, ""], minibatch_size: int
):
    """Flatten the shared environment and time axes, then shuffle and batch the leaves."""
    flat_data = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), data)
    num_samples = jax.tree.leaves(flat_data)[0].shape[0]

    num_batches = num_samples // minibatch_size

    permutation = jax.random.permutation(key, num_samples)
    shuffled_data = jax.tree.map(
        lambda x: x[permutation][: num_batches * minibatch_size], flat_data
    )
    minibatches = jax.tree.map(
        lambda x: x.reshape(num_batches, minibatch_size, *x.shape[1:]), shuffled_data
    )  # Extra dimension to match train_forward requirements
    return minibatches


def update_network(network: Network, optimizer: Optimizer, data: Any, loss_fn: Callable, config):
    (loss, info), grads = nnx.value_and_grad(loss_fn, has_aux=True)(network, data, config)
    optimizer.update(grads)
    return loss, info
