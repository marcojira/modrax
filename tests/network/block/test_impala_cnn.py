"""Tests for the IMPALA CNN block."""

import jax.numpy as jnp
import pytest

from modrax.network.block.impala_cnn import ImpalaCNN, ImpalaCNNConfig


@pytest.mark.parametrize("input_shape", [(32, 32, 3), (64, 64, 4)])
def test_impala_cnn_init_and_call(rngs, input_shape):
    """Test ImpalaCNN initialization and forward pass."""
    output_dim = 32
    batch_size = 2

    config = ImpalaCNNConfig(channels=(8, 16, 16))
    cnn = ImpalaCNN(
        input_shape=input_shape,
        output_dim=output_dim,
        config=config,
        rngs=rngs,
    )

    x = jnp.ones((batch_size, *input_shape))
    output = cnn(x)

    assert output.shape == (batch_size, output_dim)
    assert not jnp.any(jnp.isnan(output))