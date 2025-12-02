"""Test Craftax environment wrapper."""

import pytest

from modrax.env import Env, CraftaxEnvConfig
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "Craftax-Symbolic-v1",
    "Craftax-Classic-Symbolic-v1",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_craftax_env(env_name):
    """Test Craftax environment initialization, reset, and step."""
    config = CraftaxEnvConfig(env_name=env_name)
    env = Env(config, jit=False)
    run_env_test(env)
