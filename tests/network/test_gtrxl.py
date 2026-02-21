"""Tests for the GTrXL module."""

import jax.numpy as jnp
import pytest
from flax import nnx

from modrax.network.gtrxl import GatedTransformerXL


@pytest.mark.parametrize("num_layers", [2])
def test_gtrxl_eval_and_train(num_layers):
    """Test GTrXL in both eval (rollout) and train mode."""
    input_dim, output_dim, batch_size, seq_len = 16, 32, 4, 8
    num_heads = 4
    memory_len = 4
    segment_len = 4

    gtrxl = GatedTransformerXL(
        input_dim=input_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        rollout_memory_len=memory_len,
        segment_len=segment_len,
        gating=True,
        rngs=nnx.Rngs(0),
    )

    gtrxl.initialize_carry(batch_size)

    # obs = jnp.ones((batch_size, seq_len, input_dim))
    obs = jnp.ones((batch_size, input_dim))

    done = jnp.zeros((batch_size,))
    _, x = gtrxl(obs, done)
    print(x.shape)

    # # Eval (rollout) mode: single timestep
    # state = gtrxl.init_recurrent_state(batch_size)
    # obs = jnp.ones((batch_size, 1, dim))
    # y, state = gtrxl(obs, state)
    # assert y.shape == (batch_size, 1, dim)

    # # Step a few more times
    # for _ in range(3):
    #     y, state = gtrxl(obs, state)
    # assert y.shape == (batch_size, 1, dim)

    # # Reset on done
    # done = jnp.array([1, 0, 0, 0])
    # state = gtrxl.reset_recurrent_state(state, done)

    # # Train mode: full sequence
    # init_state = gtrxl.init_recurrent_state(batch_size)
    # obs_seq = jnp.ones((batch_size, seq_len, dim))
    # out = gtrxl.train_forward(obs_seq, state, init_state)
    # assert out.shape == (batch_size, seq_len, dim)
    # assert not jnp.any(jnp.isnan(out))
