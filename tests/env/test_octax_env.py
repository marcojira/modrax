"""Test Octax environment wrapper."""

import pytest

from modrax.env.octax import OctaxConfig, OctaxEnv
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "tetris",
    "pong",
    "brix",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_octax_env(env_name):
    """Test Octax environment initialization, reset, and step."""
    config = OctaxConfig(env_name=env_name)
    env = OctaxEnv(config)
    run_env_test(env)
