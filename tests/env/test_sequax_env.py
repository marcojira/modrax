"""Test Sequax environment wrapper."""

import pytest

from modrax.env.sequax import SequaxConfig, SequaxEnv
from tests.env.helpers import run_env_test

TEST_ENVS = [
    "BitSequence",
    "AMP",
]


@pytest.mark.parametrize("env_name", TEST_ENVS)
def test_sequax_env(env_name):
    """Test Sequax environment initialization, reset, and step."""
    config = SequaxConfig(env_name=env_name)
    env = SequaxEnv(config)
    run_env_test(env, num_envs=4, num_steps=18, test_render=False)  # long enough to end an episode
