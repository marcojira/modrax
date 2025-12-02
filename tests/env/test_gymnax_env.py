"""Test Gymnax environment wrapper."""

import pytest

from modrax.env import Env, GymnaxEnvConfig
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "CartPole-v1",
    "Breakout-MinAtar",
    "FourRooms-misc",
    "Catch-bsuite",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_gymnax_env(env_name):
    """Test Gymnax environment initialization, reset, and step."""
    config = GymnaxEnvConfig(env_name=env_name)
    env = Env(config, jit=False)
    run_env_test(env)
