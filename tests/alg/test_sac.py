import jax
import jax.numpy as jnp
from flax import nnx

from modrax.alg.sac import Actor, Critic, LogAlpha, SACAlg, SACConfig, SACNetwork
from tests.helpers import ContinuousCountingEnv


class TinyActor(Actor):
    def __init__(self, obs_size: int, action_size: int, rngs: nnx.Rngs):
        self.mean = nnx.Linear(obs_size, action_size, rngs=rngs)

    def get_action(self, obs, key):
        mean = self.mean(obs)
        action = jnp.tanh(mean + jax.random.normal(key, mean.shape))
        return action, jnp.zeros(obs.shape[:-1])


class TinyCritic(Critic):
    def __init__(self, obs_size: int, action_size: int, rngs: nnx.Rngs):
        self.q1 = nnx.Linear(obs_size + action_size, 1, rngs=rngs)
        self.q2 = nnx.Linear(obs_size + action_size, 1, rngs=rngs)

    def get_qs(self, obs, action):
        x = jnp.concat([obs, action], axis=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class TinySACNetwork(SACNetwork):
    def __init__(self, obs_size: int, action_size: int, rngs: nnx.Rngs):
        self.actor = TinyActor(obs_size, action_size, rngs)
        self.critic = TinyCritic(obs_size, action_size, rngs)
        self.critic_target = nnx.clone(self.critic)
        self.log_alpha = LogAlpha(0.0)
        self.running_norm = lambda obs: obs

    def policy(self, env_state, key):
        return self.actor.get_action(env_state.obs, key)


def test_sac_step_returns_finite_metrics():
    env = ContinuousCountingEnv(episode_length=2)
    network = TinySACNetwork(1, env.action_size, nnx.Rngs(0))
    cfg = SACConfig(
        total_steps=8,
        num_envs=2,
        init_buffer_size=4,
        buffer_size=64,
        minibatch_size=4,
        grad_steps=1,
        iterations_per_epoch=2,
    )
    alg = SACAlg(env, network, cfg, jax.random.key(0))
    metrics = alg.step(jax.random.key(1))

    assert all(jnp.all(jnp.isfinite(value)) for value in metrics.values())
