"""Test MuJoCo Playground environment wrapper."""

import pytest
from helpers import run_env_test

from modrax.env import Env, MuJoCoEnvConfig

TEST_ENVS = [
    "CartpoleBalance",
    "HopperStand",
    "PointMass",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_mujoco_env(env_name):
    """Test MuJoCo environment initialization, reset, and step."""
    config = MuJoCoEnvConfig(env_name=env_name)
    env = Env(config, jit=False)
    run_env_test(env, test_render=False)