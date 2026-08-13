"""Test Gymnax environment wrapper."""

import jax
import jax.numpy as jnp
import pytest
from helpers import run_env_test

from modrax.env import ContinuousActionSpec, DiscreteActionSpec
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


def test_gymnax_action_specs():
    discrete_env = GymnaxEnv(GymnaxConfig(env_name="CartPole-v1"))
    assert discrete_env.action_spec == DiscreteActionSpec(num_actions=2)
    assert discrete_env.action_size == 2

    continuous_env = GymnaxEnv(GymnaxConfig(env_name="Pendulum-v1"))
    assert isinstance(continuous_env.action_spec, ContinuousActionSpec)
    assert continuous_env.action_spec.shape == (1,)
    assert continuous_env.action_size == 1

    actions = continuous_env.sample_action(jax.random.key(0), num_envs=4)
    assert actions.shape == (4, 1)
    assert jnp.all(actions >= continuous_env.action_spec.low)
    assert jnp.all(actions <= continuous_env.action_spec.high)
