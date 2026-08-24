"""Utility functions."""

import jax
from jaxtyping import Array, Key, PyTree, Shaped


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
