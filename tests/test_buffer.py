import jax
import jax.numpy as jnp

from modrax.buffer import ReplayBuffer


def test_replay_buffer_adds_and_samples_data():
    buffer = ReplayBuffer(max_size=4)
    state = buffer.init({"obs": jnp.zeros((1, 2))})
    state = buffer.add(state, {"obs": jnp.array([[1, 2], [3, 4]])})

    batch, indices, weights = buffer.sample(state, jax.random.key(0), batch_size=3)

    assert state.size == 2
    assert batch["obs"].shape == (3, 2)
    assert jnp.all(indices < state.size)
    assert jnp.allclose(weights, 1)


def test_replay_buffer_updates_priorities():
    buffer = ReplayBuffer(max_size=3)
    state = buffer.init(jnp.zeros((1,)))

    state = buffer.update_priorities(state, jnp.array([0, 2]), jnp.array([2.0, 4.0]))

    assert jnp.array_equal(state.priorities, jnp.array([2.0, 0.0, 4.0]))
