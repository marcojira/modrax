"""Tests for the RNN block."""

import jax.numpy as jnp
import pytest

from modrax.network.block.rnn import NnxRNN, RNNConfig, RNNRecurrentState


@pytest.mark.parametrize("cell_type", ["lstm", "gru", "simple"])
def test_rnn_init_and_call(rngs, cell_type):
    """Test RNN init_recurrent_state and __call__."""
    input_dim = 16
    output_dim = 32
    batch_size = 4
    seq_len = 1

    config = RNNConfig(cell_type=cell_type)

    rnn = NnxRNN(
        input_shape=input_dim,
        output_dim=output_dim,
        config=config,
        rngs=rngs,
    )

    # Test init_recurrent_state
    state = rnn.init_recurrent_state(batch_size)
    assert isinstance(state, RNNRecurrentState)

    # Test __call__
    obs = jnp.ones((batch_size, seq_len, input_dim))
    output, new_state = rnn(obs, state)

    assert output.shape == (batch_size, seq_len, output_dim)
    assert isinstance(new_state, RNNRecurrentState)
    assert not jnp.any(jnp.isnan(output))