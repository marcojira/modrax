"""Test PGX environment wrapper."""

import pytest

from modrax.env.pgx import PGXConfig, PGXEnv
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "tic_tac_toe",
    "minatar-asterix",
    "minatar-freeway",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_pgx_env(env_name):
    """Test PGX environment initialization, reset, and step."""
    config = PGXConfig(env_name=env_name)
    env = PGXEnv(config)
    run_env_test(env, num_envs=4, num_steps=8)
