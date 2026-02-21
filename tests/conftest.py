"""Shared pytest fixtures for tests."""

import jax
import pytest
from flax import nnx

from modrax.env import Env, GymnaxEnvConfig
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


# @pytest.fixture
# def block_network_cfg():
#     """Simple BlockNetwork config for testing."""
#     return BlockNetworkConfig(
#         encoders={"obs": (MLP, MLPConfig(hidden_dims=(4,)), None)},
#         encoder_dim=4,
#         trunk=(MLP, MLPConfig(hidden_dims=(4,))),
#         trunk_dim=4,
#         heads={
#             "policy": (MLP, MLPConfig(hidden_dims=(4,)), None),
#             "value": (MLP, MLPConfig(hidden_dims=(4,)), 1),
#         },
#     )


# @pytest.fixture(params=["gtrxl", "rnn"])
# def recurrent_network_cfg(request):
#     """Parametrized RecurrentNetwork config for testing (GTrXL and RNN)."""
#     recurrent_configs = {
#         "gtrxl": (
#             GatedTransformerXL,
#             GTrXLConfig(num_heads=2, num_layers=1, rollout_memory_len=4, segment_len=4),
#         ),
#         "rnn": (NnxRNN, RNNConfig(cell_type="lstm", num_layers=2)),
#     }
#     hidden_dim = 8
#     return RecurrentNetworkConfig(
#         encoders={"obs": (MLP, MLPConfig(hidden_dims=(hidden_dim,)), None)},
#         encoder_dim=hidden_dim,
#         recurrent=recurrent_configs[request.param],
#         recurrent_dim=hidden_dim,
#         heads={
#             "policy": (MLP, MLPConfig(hidden_dims=(4,)), None),
#             "value": (MLP, MLPConfig(hidden_dims=(4,)), 1),
#         },
#     )


# @pytest.fixture
# def network(env, rngs, block_network_cfg):
#     """Simple BlockNetwork for testing."""
#     return BlockNetwork(
#         obs_shape=env.obs_shape,
#         num_actions=env.num_actions,
#         config=block_network_cfg,
#         rngs=rngs,
#     )


# @pytest.fixture
# def optimizer(network):
#     """Optimizer for testing."""
#     return Optimizer(OptimizerConfig(), network)


# @pytest.fixture
# def recurrent_optimizer(recurrent_network):
#     """Optimizer for recurrent network testing."""
#     return Optimizer(OptimizerConfig(), recurrent_network)


# @pytest.fixture
# def recurrent_network(env, rngs, recurrent_network_cfg):
#     """Simple RecurrentNetwork for testing."""
#     return RecurrentNetwork(
#         obs_shape=env.obs_shape,
#         num_actions=env.num_actions,
#         config=recurrent_network_cfg,
#         rngs=rngs,
#     )
