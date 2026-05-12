"""Test Gymnax environment wrapper."""

import pytest
from helpers import run_env_test

from modrax.env.gymnax import GymnaxConfig, GymnaxEnv

TEST_ENVS = [
    "CartPole-v1",
    "Breakout-MinAtar",
    "FourRooms-misc",
    "Catch-bsuite",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_gymnax_env(env_name):
    """Test Gymnax environment initialization, reset, and step."""
    config = GymnaxConfig(env_name=env_name)
    env = GymnaxEnv(config)
    run_env_test(env)
