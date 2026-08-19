import jax.numpy as jnp
from flax import nnx

from modrax.network.relative_attention import (
    RelativeMultiHeadSelfAttention,
    make_causal_mask,
    make_pos_emb,
)


def test_causal_mask_includes_memory_and_past_tokens():
    mask = make_causal_mask(M=2, T=3)

    assert mask.shape == (1, 1, 3, 5)
    assert jnp.array_equal(mask[0, 0, :, 2:], jnp.tril(jnp.ones((3, 3))))


def test_relative_attention_returns_one_embedding_per_token():
    attention = RelativeMultiHeadSelfAttention(
        num_heads=2,
        in_features=4,
        qkv_features=4,
        out_features=4,
        rngs=nnx.Rngs(0),
    )
    sequence = jnp.ones((2, 3, 4))
    sequence_with_memory = jnp.ones((2, 5, 4))

    output = attention(sequence, sequence_with_memory, mask=make_causal_mask(2, 3))

    assert output.shape == sequence.shape
    assert make_pos_emb(2, 5, 4).shape == (2, 5, 4)
