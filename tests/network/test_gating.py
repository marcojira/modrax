import jax.numpy as jnp
from flax import nnx

from modrax.network.gating import Gating, GatingTransformerLayer
from modrax.network.relative_attention import make_causal_mask


def test_gating_preserves_input_shape():
    gate = Gating(features=4, rngs=nnx.Rngs(0))

    output = gate(jnp.ones((2, 4)), jnp.zeros((2, 4)))

    assert output.shape == (2, 4)


def test_transformer_layer_combines_memory_and_sequence():
    layer = GatingTransformerLayer(
        num_heads=2, io_features=4, qkv_features=4, gating=True, rngs=nnx.Rngs(0)
    )

    output = layer(
        jnp.zeros((2, 2, 4)),
        jnp.ones((2, 3, 4)),
        mask=make_causal_mask(2, 3),
    )

    assert output.shape == (2, 3, 4)
