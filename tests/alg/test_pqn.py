import jax
import jax.numpy as jnp
from flax import nnx

from modrax.alg.pqn import (
    PQNAlg,
    PQNConfig,
    PQNNetwork,
    PQNNetworkOutput,
)
from modrax.policy import epsilon_greedy_policy
from tests.helpers import CountingEnv


class TinyPQNNetwork(PQNNetwork):
    def __init__(self, obs_size: int, num_actions: int, rngs: nnx.Rngs):
        self.is_recurrent = False
        self.eps = nnx.Variable(jnp.array(1.0))
        self.q_head = nnx.Linear(obs_size, num_actions, rngs=rngs)

    def train_forward(self, obs):
        return self.q_head(obs)

    def policy(self, env_state, key):
        q_values = self.train_forward(env_state.obs)
        action = epsilon_greedy_policy(q_values, env_state.action_mask, key, self.eps)
        return action, PQNNetworkOutput(q_values, None)


def test_pqn_step_returns_finite_metrics():
    env = CountingEnv(episode_length=2)
    network = TinyPQNNetwork(1, env.action_size, nnx.Rngs(0))
    cfg = PQNConfig(total_steps=32, num_envs=4, num_timesteps=4, num_updates=1, num_minibatches=2)
    alg = PQNAlg(env, network, cfg, jax.random.key(0))
    metrics = alg.step(jax.random.key(1))

    assert all(jnp.all(jnp.isfinite(value)) for value in metrics.values())
