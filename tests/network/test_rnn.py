"""Tests for the RNN module."""

import jax.numpy as jnp
import pytest
from flax import nnx

from modrax.network.rnn import NnxRNN


# @pytest.mark.parametrize("cell_type", ["lstm", "gru", "simple"])
# @pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("cell_type", ["lstm"])
@pytest.mark.parametrize("num_layers", [2])
def test_rnn_train_and_eval(cell_type, num_layers):
    """Test RNN in both eval and train mode."""
    input_dim, output_dim, batch_size, seq_len = 16, 32, 4, 8

    rnn = NnxRNN(
        input_dim=input_dim,
        output_dim=output_dim,
        rngs=nnx.Rngs(0),
        cell_type=cell_type,
        num_layers=num_layers,
    )
    rnn.initialize_carry(batch_size)

    # obs = jnp.ones((batch_size, seq_len, input_dim))
    obs = jnp.ones((batch_size, input_dim))

    done = jnp.zeros((batch_size,))
    _, x = rnn(obs, done)
    # print(x.shape)

    rnn.train()
    obs = jnp.ones((batch_size, seq_len, input_dim))
    done = jnp.zeros((batch_size, seq_len))
    x = rnn(obs, done)
    print(x.shape)

    # # Eval mode
    # rnn.eval()
    # out = rnn(obs)
    # assert out.shape == (batch_size, seq_len, output_dim)
    # assert not jnp.any(jnp.isnan(out))

    # # Train mode
    # rnn.train()
    # rnn.initialize_carry(batch_size)
    # dones = jnp.zeros((batch_size, seq_len))
    # out = rnn(obs, dones)
    # assert out.shape == (batch_size, seq_len, output_dim)
    # assert not jnp.any(jnp.isnan(out))
