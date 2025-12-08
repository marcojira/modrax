"""Shared pytest fixtures for tests."""

import jax
import pytest
from flax import nnx

from modrax.env import Env, GymnaxEnvConfig
from modrax.network.block.mlp import MLP, MLPConfig
from modrax.network.block_network import BlockNetwork, BlockNetworkConfig
from modrax.optimizer import Optimizer, OptimizerConfig


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


@pytest.fixture
def network(env, rngs):
    """Simple BlockNetwork for testing."""
    config = BlockNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(4,)))},
        encoder_dim=4,
        trunk=(MLP, MLPConfig(hidden_dims=(4,))),
        trunk_dim=4,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(4,))),
            "value": (MLP, MLPConfig(hidden_dims=(4,))),
        },
    )
    return BlockNetwork(
        input_shapes={"obs": env.obs_shape},
        output_dims={"policy": env.num_actions, "value": 1},
        config=config,
        rngs=rngs,
    )


@pytest.fixture
def optimizer(network):
    """Optimizer for testing."""
    return Optimizer(OptimizerConfig(), network)
