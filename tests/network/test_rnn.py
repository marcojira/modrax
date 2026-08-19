import jax
import jax.numpy as jnp
from flax import nnx

from modrax.network.rnn import NnxRNN


def test_rnn_tracks_and_resets_episode_carry():
    rnn = NnxRNN(3, 4, nnx.Rngs(0), cell_type="gru")
    rnn.initialize_carry(batch_size=2)

    _, output = rnn(jnp.ones((2, 3)))
    rnn.reset_episodes(jnp.array([True, False]))

    assert output.shape == (2, 4)
    carry = jax.tree.leaves(rnn.carry.get_value())[0]
    assert jnp.allclose(carry[0], 0)
    assert not jnp.allclose(carry[1], 0)
