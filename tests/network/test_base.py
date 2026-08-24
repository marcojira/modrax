import jax.numpy as jnp
from flax import nnx

from modrax.network import Network


class _Network(Network):
    def __init__(self):
        self.dropout = nnx.Dropout(0.5)
        self.state = nnx.Variable(jnp.ones(2))

    def reset(self):
        self.state[...] = 0


def test_eval_sets_layer_mode_and_resets_state():
    network = _Network()

    network.eval()

    assert network.dropout.deterministic
    assert jnp.array_equal(network.state[...], jnp.zeros(2))
