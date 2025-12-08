"""Test PGX environment wrapper."""

import pytest
from helpers import run_env_test

from modrax.env import Env, PGXEnvConfig

TEST_ENVS = [
    "minatar-asterix",
    "minatar-breakout",
    "minatar-freeway",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_pgx_env(env_name):
    """Test PGX environment initialization, reset, and step."""
    config = PGXEnvConfig(env_name=env_name)
    env = Env(config, jit=False)
    run_env_test(env, num_envs=4, num_steps=8)
