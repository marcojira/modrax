import jax.numpy as jnp
from flax import nnx

from modrax.network.mlp import MLP


def test_mlp_maps_the_last_dimension():
    network = MLP(3, [4], 2, jnp.tanh, nnx.Rngs(0))

    output = network(jnp.ones((2, 5, 3), dtype=jnp.int32))

    assert output.shape == (2, 5, 2)
    assert output.dtype == jnp.float32
