"""Shared pytest fixtures for tests."""

import jax
import pytest
from flax import nnx

from modrax.env import Env, GymnaxEnvConfig
from modrax.network.block.gtrxl import GatedTransformerXL, GTrXLConfig
from modrax.network.block.mlp import MLP, MLPConfig
from modrax.network.block.rnn import NnxRNN, RNNConfig
from modrax.network.block_network import BlockNetwork, BlockNetworkConfig
from modrax.network.recurrent_network import RecurrentNetwork, RecurrentNetworkConfig
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


@pytest.fixture
def recurrent_optimizer(recurrent_network):
    """Optimizer for recurrent network testing."""
    return Optimizer(OptimizerConfig(), recurrent_network)


@pytest.fixture(params=["gtrxl", "rnn"])
def recurrent_network(request, env, rngs):
    """Parametrized recurrent network fixture (GTrXL and RNN)."""
    recurrent_configs = {
        "gtrxl": (GatedTransformerXL, GTrXLConfig(
            num_heads=2, num_layers=1, rollout_memory_len=4, segment_len=4
        )),
        "rnn": (NnxRNN, RNNConfig(cell_type="lstm")),
    }

    hidden_dim = 8
    config = RecurrentNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(hidden_dim,)))},
        encoder_dim=hidden_dim,
        recurrent=recurrent_configs[request.param],
        recurrent_dim=hidden_dim,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(4,))),
            "value": (MLP, MLPConfig(hidden_dims=(4,))),
        },
    )
    return RecurrentNetwork(
        input_shapes={"obs": env.obs_shape},
        output_dims={"policy": env.num_actions, "value": 1},
        config=config,
        rngs=rngs,
    )
