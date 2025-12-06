"""Test RelativeMultiHeadSelfAttention module."""

import jax.numpy as jnp
import pytest
from flax import nnx

from modrax.network.module.relative_attention import RelativeMultiHeadSelfAttention


@pytest.mark.parametrize("use_mask", [False, True])
def test_relative_attention(use_mask):
    """Test relative attention forward pass with and without mask."""
    batch_size = 2
    T = 4  # Current sequence length
    M = 3  # Memory length
    num_heads = 2
    in_features = 16
    qkv_features = 8
    out_features = 16

    attn = RelativeMultiHeadSelfAttention(
        num_heads=num_heads,
        in_features=in_features,
        qkv_features=qkv_features,
        out_features=out_features,
        rngs=nnx.Rngs(0),
    )

    # Create inputs
    E = jnp.ones((batch_size, T, in_features))
    E_tilde = jnp.ones((batch_size, M + T, in_features))
    mask = jnp.ones((batch_size, num_heads, T, M + T)) if use_mask else None

    output = attn(E, E_tilde, mask=mask)

    assert output.shape == (batch_size, T, out_features)
    assert not jnp.any(jnp.isnan(output))
