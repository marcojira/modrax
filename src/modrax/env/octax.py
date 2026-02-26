"""Wrapper for Octax, from https://github.com/riiswa/octax"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key
from octax.environments import create_environment  # type: ignore

from modrax.env.base import Env, EnvConfig, State, StateWithMetrics, StepOutput


@dataclass
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
    def __init__(self, config: OctaxConfig, jit: bool = True):
        self._env, self._metadata = create_environment(config.env_name)

        # Get observation shape by doing a dummy reset
        import jax

        dummy_key = jax.random.key(0)
        _, dummy_obs, _ = self._env.reset(dummy_key)
        self.obs_shape = dummy_obs.shape
        self.action_size = self._env.num_actions

        # Call parent init to setup functions
        super().__init__(config, jit=jit)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        env_state, obs, info = self._env.reset(key)

        return State(
            env_state=env_state,
            obs=obs,
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
            obs=next_obs,
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
        )

        return step_output, new_state

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        """
        Render Octax environment observation.
        Converts boolean observation to RGB image where False=white and True=black.
        For frameskipped observations (4 x W x H), takes the most recent frame.
        """
        obs = state.obs

        # If observation has multiple frames (frameskip), take the most recent one
        if obs.ndim == 3:
            obs = obs[-1]  # Take last frame: (W, H)

        # Transpose to (H, W) for proper display
        obs = obs.T

        # Convert boolean to grayscale: False -> 255 (white), True -> 0 (black)
        grayscale = np.where(obs, 0, 255).astype(np.uint8)

        rgb_array = np.stack([grayscale, grayscale, grayscale], axis=-1)

        return rgb_array
