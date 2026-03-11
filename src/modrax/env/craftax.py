"""Wrapper for Craftax and Craftax-Classic from https://github.com/MichaelTMatthews/Craftax"""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax.constants import Achievement
from craftax.craftax_env import make_craftax_env_from_name
from jaxtyping import Array, Key

from modrax.env.base import Env, EnvConfig, State, StateWithMetrics, StepOutput


@dataclass(frozen=True)
class CraftaxConfig(EnvConfig):
    env_name: Literal[
        "Craftax-Symbolic-v1",
        "Craftax-Pixels-v1",
        "Craftax-Classic-Symbolic-v1",
        "Craftax-Classic-Pixels-v1",
    ] = "Craftax-Symbolic-v1"


class CraftaxEnv(Env):
    def __init__(self, config: CraftaxConfig):
        self._env = make_craftax_env_from_name(config.env_name, auto_reset=False)
        self._env_params = self._env.default_params

        # Get observation and action shapes from environment spaces
        self.obs_shape = self._env.observation_space(self._env_params).shape  # type: ignore
        self.action_size = self._env.action_space(self._env_params).n  # type: ignore

        super().__init__(config)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        obs, craftax_state = self._env.reset(key, self._env_params)

        return State(
            env_state=craftax_state,
            obs=obs,
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
            info=self._achievements_info(craftax_state),
        )

    def _inner_step_fn(
        self, state: State, action: Array, key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        obs, craftax_state, reward, done, info = self._env.step(
            key, state.env_state, action, self._env_params
        )

        step_output = StepOutput(
            reward=reward, done=done.astype(bool), truncation=jnp.bool_(False), info=info
        )
        new_state = State(
            env_state=craftax_state,
            obs=obs,
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
            info=self._achievements_info(craftax_state),
        )

        return step_output, new_state

    @staticmethod
    def _achievements_info(craftax_state) -> dict:
        achievements = craftax_state.achievements * 100.0
        return {a.name.lower(): achievements[a.value] for a in Achievement}

    def _get_renderer(self):
        if "Classic" in self.config.env_name:
            from craftax.craftax_classic.renderer import render_craftax_pixels
        else:
            from craftax.craftax.renderer import render_craftax_pixels
        return render_craftax_pixels

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        rgb_array = self._get_renderer()(state.env_state, block_pixel_size=16)
        return np.array(rgb_array, dtype=np.uint8)

    def batch_render(self, states: StateWithMetrics) -> np.ndarray:
        frames = jax.vmap(self._get_renderer(), in_axes=(0, None))(states.env_state, 16)
        return np.array(frames, dtype=np.uint8)
