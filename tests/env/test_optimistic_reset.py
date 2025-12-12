"""Test optimistic resets for PGX environments."""

from helpers import run_env_test

from modrax.env import Env, PGXEnvConfig


def test_pgx_optimistic_reset():
    """Test PGX environment with optimistic resets enabled."""
    config = PGXEnvConfig(
        env_name="minatar-asterix",
        optimistic_reset=True,
        num_reset_envs=4,
    )
    env = Env(config, jit=False)
    run_env_test(env, num_envs=8, num_steps=16, test_render=False)