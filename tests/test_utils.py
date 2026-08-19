import jax
import jax.numpy as jnp

from modrax.utils import batch_trajectories, batch_transitions


def test_batch_trajectories_preserves_time_axis():
    trajectories = jnp.arange(24).reshape(4, 3, 2)

    batches = batch_trajectories(trajectories, jax.random.key(0), minibatch_size=2)

    assert batches.shape == (2, 2, 3, 2)
    assert jnp.array_equal(jnp.sort(batches.reshape(-1)), jnp.arange(24))


def test_batch_transitions_flattens_environment_and_time():
    transitions = jnp.arange(24).reshape(4, 3, 2)

    batches = batch_transitions(transitions, jax.random.key(0), minibatch_size=4)

    assert batches.shape == (3, 4, 2)
    assert jnp.array_equal(jnp.sort(batches.reshape(-1)), jnp.arange(24))
