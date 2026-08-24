"""Test MuJoCo Playground environment wrapper."""

import pytest

from modrax.env.mujoco_playground import MuJoCoPlaygroundConfig, MuJoCoPlaygroundEnv
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "CartpoleBalance",
    "HopperStand",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_mujoco_env(env_name):
    """Test MuJoCo environment initialization, reset, and step."""
    config = MuJoCoPlaygroundConfig(env_name=env_name)
    env = MuJoCoPlaygroundEnv(config)
    run_env_test(env, test_render=False)
