import jax
import jax.numpy as jnp

from tests.helpers import ContinuousCountingEnv, CountingEnv


def _step(env, state, num_steps):
    key = jax.random.key(1)
    num_envs = state.obs.shape[0]
    for _ in range(num_steps):
        key, step_key = jax.random.split(key)
        action = jnp.zeros(num_envs, jnp.int32)
        output, state = env.step(state, action, jax.random.split(step_key, num_envs))
    return output, state


def test_reset_starts_episode_metrics_at_zero():
    state = CountingEnv().reset(jax.random.split(jax.random.key(0), 3))

    assert state.obs.shape == (3, 1)
    assert jnp.array_equal(state.episode_return, jnp.zeros(3))
    assert jnp.array_equal(state.episode_length, jnp.zeros(3, jnp.int32))


def test_auto_reset_restarts_the_episode_once_done():
    env = CountingEnv(episode_length=2)
    state = env.reset(jax.random.split(jax.random.key(0), 2))

    output, state = _step(env, state, num_steps=1)
    assert not output.done.any()
    assert jnp.array_equal(state.obs[:, 0], jnp.ones(2))
    assert jnp.array_equal(state.episode_return, jnp.ones(2))

    output, state = _step(env, state, num_steps=1)
    assert output.done.all()
    assert jnp.array_equal(state.obs[:, 0], jnp.zeros(2))
    assert jnp.array_equal(state.episode_return, jnp.zeros(2))
    assert jnp.array_equal(state.episode_length, jnp.zeros(2, jnp.int32))


def test_optimistic_reset_restarts_the_episode_from_a_precomputed_reset():
    env = CountingEnv(episode_length=2, optimistic_reset=True, num_reset_envs=4)
    state = env.reset(jax.random.split(jax.random.key(0), 2))

    _, state = _step(env, state, num_steps=2)

    assert jnp.array_equal(state.obs[:, 0], jnp.zeros(2))
    assert jnp.array_equal(state.episode_return, jnp.zeros(2))


def test_without_auto_reset_episode_metrics_keep_accumulating():
    env = CountingEnv(episode_length=2, auto_reset=False)
    state = env.reset(jax.random.split(jax.random.key(0), 2))

    _, state = _step(env, state, num_steps=3)

    assert jnp.array_equal(state.obs[:, 0], jnp.full(2, 3.0))
    assert jnp.array_equal(state.episode_return, jnp.full(2, 3.0))
    assert jnp.array_equal(state.episode_length, jnp.full(2, 3, jnp.int32))


def test_sample_action_matches_the_action_spec():
    discrete_env = CountingEnv()
    continuous_env = ContinuousCountingEnv()

    discrete_actions = discrete_env.sample_action(jax.random.key(0), num_envs=4)
    continuous_actions = continuous_env.sample_action(jax.random.key(0), num_envs=4)

    assert discrete_env.action_size == 2
    assert discrete_actions.shape == (4,)
    assert jnp.all((discrete_actions >= 0) & (discrete_actions < 2))

    assert continuous_env.action_size == 2
    assert continuous_actions.shape == (4, 2)
    assert jnp.all(jnp.abs(continuous_actions) <= 1)
