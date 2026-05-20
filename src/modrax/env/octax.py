"""Wrapper for Octax, from https://github.com/riiswa/octax"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key
from octax.environments import create_environment  # type: ignore

from modrax.env.base import Env, EnvConfig, State, StateWithMetrics, StepOutput


@dataclass(frozen=True)
class OctaxConfig(EnvConfig):
    """Configuration for Octax environments."""

    env_name: Literal[
        "airplane",
        "blinky",
        "brix",
        "deep",
        "filter",
        "flight_runner",
        "missile",
        "pong",
        "rocket",
        "shooting_stars",
        "spacejam",
        "squash",
        "submarine",
        "tank",
        "tetris",
        "ufo",
        "vertical_brix",
        "wipe_off",
        "worm",
    ] = "brix"


class OctaxEnv(Env):
    def __init__(self, config: OctaxConfig):
        self._env, self._metadata = create_environment(config.env_name)

        # Get observation shape by doing a dummy reset
        import jax

        dummy_key = jax.random.key(0)
        _, dummy_obs, _ = self._env.reset(dummy_key)
        self.obs_shape = self._transpose_obs(dummy_obs).shape
        self.action_size = self._env.num_actions

        super().__init__(config)

    @staticmethod
    def _transpose_obs(obs: Array) -> Array:
        # (frames, W, H) -> (H, W, frames) so frame stack acts as channels
        # under the (H, W, C) layout the networks expect.
        return jnp.transpose(obs, (2, 1, 0))

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        env_state, obs, info = self._env.reset(key)

        return State(
            env_state=env_state,
            obs=self._transpose_obs(obs),
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
        )

    def _inner_step_fn(
        self, state: State, action: Array, key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        next_env_state, next_obs, reward, terminated, truncated, info = self._env.step(
            state.env_state, action
        )

        done = jnp.logical_or(terminated, truncated).astype(jnp.bool)

        step_output = StepOutput(
            reward=reward, done=done, truncation=truncated.astype(jnp.bool), info=info
        )
        new_state = State(
            env_state=next_env_state,
            obs=self._transpose_obs(next_obs),
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
        )

        return step_output, new_state

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        """
        Render Octax environment observation.
        Converts boolean observation to RGB image where False=white and True=black.
        For frame-stacked observations (H x W x 4), takes the most recent frame.
        """
        obs = np.array(state.obs)

        if obs.ndim == 3:
            obs = obs[..., -1]  # Take last frame: (H, W)

        grayscale = np.where(obs, 0, 255).astype(np.uint8)
        return np.stack([grayscale, grayscale, grayscale], axis=-1)

    def batch_render(self, states: StateWithMetrics) -> np.ndarray:
        obs = np.array(states.obs)  # (B, H, W, 4)

        if obs.ndim == 4:
            obs = obs[..., -1]  # (B, H, W)

        grayscale = np.where(obs, 0, 255).astype(np.uint8)
        return np.stack([grayscale, grayscale, grayscale], axis=-1)  # (B, H, W, 3)
