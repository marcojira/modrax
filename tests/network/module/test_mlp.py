"""Test MLP module."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from modrax.network.module.mlp import MLP


@pytest.mark.parametrize(
    "input_shape,batch_size",
    [
        (10, 4),  # Flat input
        ((8, 8, 3), 2),  # Multi-dimensional input
    ],
)
def test_mlp(input_shape, batch_size):
    """Test MLP forward pass with flat and multi-dimensional inputs."""
    hidden_dims = [64, 32]
    output_dim = 5

    mlp = MLP(
        input_shape=input_shape,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation_fn=jax.nn.relu,
        rngs=nnx.Rngs(0),
    )

    # Create input
    if isinstance(input_shape, int):
        x = jnp.ones((batch_size, input_shape))
    else:
        x = jnp.ones((batch_size, *input_shape))

    output = mlp(x)

    assert output.shape == (batch_size, output_dim)
    assert not jnp.any(jnp.isnan(output))
