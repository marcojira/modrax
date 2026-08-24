import jax.numpy as jnp
from flax import nnx

from modrax.network.gtrxl import GatedTransformerXL


def test_gtrxl_updates_and_resets_episode_memory():
    network = GatedTransformerXL(
        input_dim=4,
        num_heads=2,
        num_layers=1,
        rollout_memory_len=2,
        segment_len=2,
        rngs=nnx.Rngs(0),
    )
    network.initialize_carry(batch_size=2)

    _, output = network(jnp.ones((2, 4)))
    network.reset_episodes(jnp.array([True, False]))

    assert output.shape == (2, 4)
    assert jnp.allclose(network.carry.memory[0], 0)
    assert not jnp.allclose(network.carry.memory[1], 0)
