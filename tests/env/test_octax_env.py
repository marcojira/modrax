"""Test Octax environment wrapper."""

import pytest
from helpers import run_env_test

from modrax.env.octax import OctaxConfig, OctaxEnv

TEST_ENVS = [
    "tetris",
    "pong",
    "brix",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_octax_env(env_name):
    """Test Octax environment initialization, reset, and step."""
    config = OctaxConfig(env_name=env_name)
    env = OctaxEnv(config, jit=False)
    run_env_test(env)
