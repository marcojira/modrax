import jax
import jax.numpy as jnp

from modrax.rollout import trajectory_rollout, trajectory_to_transitions
from tests.helpers import ConstantPolicy, CountingEnv


def test_trajectory_rollout_is_batch_major():
    env = CountingEnv(episode_length=10, auto_reset=False)
    state = env.reset(jax.random.split(jax.random.key(0), 2))

    final_state, trajectory = trajectory_rollout(
        ConstantPolicy(), env.step, state, num_steps=3, key=jax.random.key(1)
    )
    transitions = trajectory_to_transitions(trajectory, final_state)

    assert trajectory.obs.shape == (2, 3, 1)
    assert jnp.array_equal(trajectory.obs[0, :, 0], jnp.array([0, 1, 2]))
    assert jnp.array_equal(transitions.next_obs[0, :, 0], jnp.array([1, 2, 3]))
