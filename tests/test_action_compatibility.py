import jax
import jax.numpy as jnp
import pytest

from modrax.alg.ppo import PPOAlg, PPOConfig
from modrax.alg.pqn import PQNAlg, PQNConfig
from modrax.alg.sac import SACAlg, SACConfig
from modrax.env import ContinuousActionSpec, DiscreteActionSpec


class _Env:
    def __init__(self, action_spec):
        self.action_spec = action_spec


def test_discrete_algorithms_reject_continuous_actions():
    env = _Env(ContinuousActionSpec(shape=(2,), low=-jnp.ones(2), high=jnp.ones(2)))
    key = jax.random.key(0)

    with pytest.raises(ValueError, match="PPO requires a discrete action space"):
        PPOAlg(env, None, PPOConfig(), key)

    with pytest.raises(ValueError, match="PQN requires a discrete action space"):
        PQNAlg(env, None, PQNConfig(), key)


def test_sac_rejects_discrete_actions():
    env = _Env(DiscreteActionSpec(num_actions=2))

    with pytest.raises(ValueError, match="SAC requires a continuous action space"):
        SACAlg(env, None, SACConfig(), jax.random.key(0))
