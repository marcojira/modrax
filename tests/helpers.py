"""Fake environments and data builders shared across tests."""

import jax.numpy as jnp

from modrax.env.base import (
    ContinuousActionSpec,
    DiscreteActionSpec,
    Env,
    EnvConfig,
    State,
    StepOutput,
)
from modrax.network.base import Network
from modrax.rollout import Trajectory


class CountingEnv(Env):
    """Env whose observation is the step count and that rewards 1 per step."""

    obs_shape = (1,)
    action_spec = DiscreteActionSpec(num_actions=2)

    def __init__(self, episode_length: int = 2, **config):
        super().__init__(EnvConfig(**config))
        self.episode_length = episode_length

    def _inner_reset_fn(self, key):
        return State(jnp.zeros((), jnp.int32), jnp.zeros(1), self._action_mask(), {})

    def _inner_step_fn(self, state, action, key):
        step = state.env_state + 1
        output = StepOutput(jnp.ones(()), step >= self.episode_length, jnp.bool_(False), {})
        obs = jnp.full((1,), step, jnp.float32)
        return output, State(step, obs, state.action_mask, {})

    def _action_mask(self):
        return jnp.ones(self.action_size, jnp.bool_)


class ContinuousCountingEnv(CountingEnv):
    """CountingEnv with a 2D continuous action space."""

    action_spec = ContinuousActionSpec(shape=(2,), low=-jnp.ones(2), high=jnp.ones(2))


class ConstantPolicy(Network):
    """Network that always selects the same action."""

    def __init__(self, action: int = 0):
        self.action = action

    def policy(self, env_state, key):
        return jnp.full(env_state.obs.shape[0], self.action, jnp.int32), None


def make_trajectory(rewards, **overrides) -> Trajectory:
    """Build a Trajectory from (B, T) rewards, zero-filling every field not overridden."""
    shape = rewards.shape
    fields = dict(
        obs=rewards[..., None],
        info={},
        actions=jnp.zeros(shape, jnp.int32),
        rewards=rewards,
        action_masks=jnp.ones((*shape, 1), jnp.bool_),
        network_output=None,
        dones=jnp.zeros(shape, jnp.bool_),
        truncations=jnp.zeros(shape, jnp.bool_),
        valid_mask=jnp.ones(shape, jnp.bool_),
        episode_returns=jnp.zeros(shape),
        episode_lengths=jnp.zeros(shape),
    )
    return Trajectory(**(fields | overrides))
