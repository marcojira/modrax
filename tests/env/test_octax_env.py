"""Test Octax environment wrapper."""

import pytest

from modrax.env import Env, OctaxEnvConfig
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "tetris",
    "pong",
    "brix",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_octax_env(env_name):
    """Test Octax environment initialization, reset, and step."""
    config = OctaxEnvConfig(env_name=env_name)
    env = Env(config, jit=False)
    run_env_test(env)
