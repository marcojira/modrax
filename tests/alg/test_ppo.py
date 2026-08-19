import jax
import jax.numpy as jnp
from flax import nnx

from modrax.alg.ppo import (
    PPOAlg,
    PPOConfig,
    PPONetwork,
    PPONetworkOutput,
)
from modrax.policy import softmax_policy
from tests.helpers import CountingEnv


class TinyPPONetwork(PPONetwork):
    def __init__(self, obs_size: int, num_actions: int, rngs: nnx.Rngs):
        self.policy_head = nnx.Linear(obs_size, num_actions, rngs=rngs)
        self.value_head = nnx.Linear(obs_size, 1, rngs=rngs)

    def train_forward(self, obs, dones, init_carry, saved_carry):
        return PPONetworkOutput(self.policy_head(obs), self.value_head(obs), None)

    def policy(self, env_state, key):
        output = self.train_forward(env_state.obs, None, None, None)
        return softmax_policy(output.policy, env_state.action_mask, key), output


def test_ppo_step_returns_finite_metrics():
    env = CountingEnv(episode_length=2)
    network = TinyPPONetwork(1, env.action_size, nnx.Rngs(0))
    cfg = PPOConfig(total_steps=32, num_envs=4, num_gen_steps=4, num_updates=1, num_minibatches=2)
    alg = PPOAlg(env, network, cfg, jax.random.key(0))
    metrics = alg.step(jax.random.key(1))

    assert all(jnp.all(jnp.isfinite(value)) for value in metrics.values())
