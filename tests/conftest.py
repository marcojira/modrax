"""Shared pytest fixtures for tests."""

import jax
import pytest
from flax import nnx

from modrax.env import Env, GymnaxEnvConfig


@pytest.fixture
def rng_key():
    """Basic JAX random key."""
    return jax.random.key(0)


@pytest.fixture
def env():
    """Simple CartPole environment for testing networks."""
    config = GymnaxEnvConfig(env_name="CartPole-v1")
    return Env(config, jit=False)


@pytest.fixture
def rngs():
    """Random number generator for network initialization."""
    return nnx.Rngs(0)


@pytest.fixture
def batch_obs(env):
    """Batch of sample observations from the environment."""
    keys = jax.random.split(jax.random.key(0), 4)
    state = env.reset(keys)
    return state.obs
