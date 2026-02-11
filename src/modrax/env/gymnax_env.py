"""Wrapper for Gymnax environments from https://github.com/RobertTLange/gymnax"""

from typing import Literal

import gymnax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Key

from modrax.env.base import Env, EnvConfig, State, StateWithMetrics, StepOutput
from modrax.utils import fig_to_rgb_array


class GymnaxEnvConfig(EnvConfig):
    env_name: Literal[
        # Classic Control
        "Acrobot-v1",
        "CartPole-v1",
        "MountainCar-v0",
        "MountainCarContinuous-v0",
        "Pendulum-v1",
        # MinAtar
        "Asterix-MinAtar",
        "Breakout-MinAtar",
        "Freeway-MinAtar",
        "SpaceInvaders-MinAtar",
        # BSuite
        "Catch-bsuite",
        "DeepSea-bsuite",
        "DiscountingChain-bsuite",
        "MemoryChain-bsuite",
        "UmbrellaChain-bsuite",
        "MNISTBandit-bsuite",
        "SimpleBandit-bsuite",
        # Misc
        "BernoulliBandit-misc",
        "GaussianBandit-misc",
        "FourRooms-misc",
        "MetaMaze-misc",
        "PointRobot-misc",
        "Pong-misc",
        "Reacher-misc",
        "Swimmer-misc",
    ] = "CartPole-v1"


class GymnaxEnv(Env):
    def __init__(self, config: GymnaxEnvConfig, jit: bool = True):
        self._env, self._env_params = gymnax.make(config.env_name)

        # Get observation and action shapes from environment spaces
        self.obs_shape = self._env.observation_space(self._env_params).shape  # type: ignore
        self.num_actions = self._env.action_space(self._env_params).n  # type: ignore

        # Call parent init to setup functions
        super().__init__(config, jit=jit)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        obs, gymnax_state = self._env.reset(key, self._env_params)

        return State(
            env_state=gymnax_state,
            obs=obs.astype(jnp.int8),
            action_mask=jnp.ones(self.num_actions, dtype=jnp.float32),
        )

    def _inner_step_fn(
        self, state: State, action: Array, key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        obs, gymnax_state, reward, done, info = self._env.step(
            key, state.env_state, action, self._env_params
        )

        step_output = StepOutput(reward=reward, done=done, truncation=jnp.bool_(False), info=info)
        new_state = State(
            env_state=gymnax_state,
            obs=obs.astype(jnp.int8),
            action_mask=jnp.ones(self.num_actions, dtype=jnp.float32),
        )

        return step_output, new_state

    """ Using logic from https://github.com/RobertTLange/gymnax/blob/main/gymnax/visualize/"""

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        if self.config.env_name in [
            "Acrobot-v1",
            "CartPole-v1",
            "Pendulum-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
        ]:
            import gymnasium as gym
            from gymnax.visualize.vis_gym import get_gym_state, set_gym_params

            if self.config.env_name == "Pendulum-v1":
                gym_env = gym.make("Pendulum-v0", render_mode="rgb_array")
            else:
                gym_env = gym.make(self.config.env_name, render_mode="rgb_array")
            gym_env.reset()
            set_gym_params(gym_env, self.config.env_name, self._env_params)
            gym_state = get_gym_state(state.env_state, self.config.env_name)
            if self.config.env_name == "Pendulum-v1":
                gym_env.env.last_u = gym_state[-1]  # type: ignore
            gym_env.env.state = gym_state  # type: ignore
            rgb_array = gym_env.render()
            gym_env.close()
            return rgb_array  # type: ignore
        elif self.config.env_name == "Catch-bsuite":
            from gymnax.visualize import vis_catch

            fig, ax = plt.subplots()
            im = vis_catch.init_catch(ax, self._env, state.env_state, self._env_params)
            vis_catch.update_catch(im, self._env, state.env_state)
            return fig_to_rgb_array(fig)
        elif self.config.env_name in [
            "Asterix-MinAtar",
            "Breakout-MinAtar",
            "Freeway-MinAtar",
            "Seaquest-MinAtar",
            "SpaceInvaders-MinAtar",
            "Pong-misc",
        ]:
            from gymnax.visualize import vis_minatar

            fig, ax = plt.subplots()
            im = vis_minatar.init_minatar(ax, self._env, state.env_state)
            vis_minatar.update_minatar(im, self._env, state.env_state)
            return fig_to_rgb_array(fig)
        elif self.config.env_name == "PointRobot-misc":
            from gymnax.visualize import vis_circle

            fig, ax = plt.subplots()
            im = vis_circle.init_circle(ax, self._env, state.env_state, self._env_params)
            vis_circle.update_circle(im, self._env, state.env_state)
            return fig_to_rgb_array(fig)
        elif self.config.env_name in ["MetaMaze-misc", "FourRooms-misc"]:
            from gymnax.visualize import vis_maze

            fig, ax = plt.subplots()
            im = vis_maze.init_maze(ax, self._env, state.env_state, self._env_params)
            vis_maze.update_maze(im, self._env, state.env_state)
            return fig_to_rgb_array(fig)
        else:
            raise NotImplementedError(f"Rendering not implemented for {self.config.env_name}")
