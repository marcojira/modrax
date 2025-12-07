"""Tests for the GTrXL block."""

import jax.numpy as jnp

from modrax.network.block.gtrxl import GatedTransformerXL, GTrXLConfig, GTrXLRecurrentState


def test_gtrxl_init_and_call(rngs):
    """Test GTrXL init_recurrent_state and __call__."""
    input_dim = 16
    batch_size = 4
    seq_len = 1

    config = GTrXLConfig(
        num_heads=2,
        num_layers=2,
        rollout_memory_len=4,
        segment_len=2,
    )

    gtrxl = GatedTransformerXL(
        input_shape=input_dim,
        output_dim=input_dim,
        config=config,
        rngs=rngs,
    )

    # Test init_recurrent_state
    state = gtrxl.init_recurrent_state(batch_size)
    assert isinstance(state, GTrXLRecurrentState)
    assert state.memory.shape == (batch_size, config.rollout_memory_len, config.num_layers, input_dim)
    assert state.mask.shape == (batch_size, 1, 1, 1 + config.rollout_memory_len)

    # Test __call__
    encoded_obs = jnp.ones((batch_size, seq_len, input_dim))
    output, new_state = gtrxl(encoded_obs, state)

    assert output.shape == encoded_obs.shape
    assert isinstance(new_state, GTrXLRecurrentState)
    assert not jnp.any(jnp.isnan(output))