"""Shared test helpers for environment tests."""

import jax


def run_env_test(env, num_envs=2, num_steps=4, test_render=True):
    """Run standard environment test: reset, step, and verify outputs.

    Args:
        env: Environment instance to test
        num_envs: Number of parallel environments
        num_steps: Number of steps to run
        test_render: Whether to test rendering functionality

    Returns:
        Final state after all steps
    """
    key = jax.random.key(0)
    state = env.reset(jax.random.split(key, num_envs))

    for _ in range(num_steps):
        key, step_key = jax.random.split(key)
        action = env.sample_action(step_key, num_envs)
        out, state = env.step(state, action, jax.random.split(key, num_envs))

    # Verify StepOutput properties
    assert out.reward.shape == (num_envs,)
    assert out.done.shape == (num_envs,)
    assert out.info is not None

    # Verify State properties
    assert state.obs.shape[0] == num_envs
    assert state.episode_length.shape == (num_envs,)
    assert state.episode_return.shape == (num_envs,)

    if test_render:
        single_state = jax.tree.map(lambda x: x[0], state)
        frame = env.render(single_state)
        assert frame.shape[-1] == 3
        assert len(frame.shape) == 3

    return state
