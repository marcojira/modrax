"""Wrapper for MuJoCo Playground environments from https://github.com/google-deepmind/mujoco_playground"""

import os
from dataclasses import dataclass
from typing import Literal

os.environ.setdefault("MUJOCO_GL", "egl")

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Key

from modrax.env.base import Env, EnvConfig, State, StateWithMetrics, StepOutput


@dataclass
class MuJoCoPlaygroundConfig(EnvConfig):
    env_name: Literal[
        # DM Control Suite
        "AcrobotSwingup",
        "AcrobotSwingupSparse",
        "BallInCup",
        "CartpoleBalance",
        "CartpoleBalanceSparse",
        "CartpoleSwingup",
        "CartpoleSwingupSparse",
        "CheetahRun",
        "FingerSpin",
        "FingerTurnEasy",
        "FingerTurnHard",
        "FishSwim",
        "HopperHop",
        "HopperStand",
        "HumanoidRun",
        "HumanoidStand",
        "HumanoidWalk",
        "PendulumSwingup",
        "PointMass",
        "ReacherEasy",
        "ReacherHard",
        "SwimmerSwimmer6",
        "WalkerRun",
        "WalkerStand",
        "WalkerWalk",
        # Locomotion
        "ApolloJoystickFlatTerrain",
        "BarkourJoystick",
        "BerkeleyHumanoidJoystickFlatTerrain",
        "BerkeleyHumanoidJoystickRoughTerrain",
        "G1JoystickFlatTerrain",
        "G1JoystickRoughTerrain",
        "Go1Footstand",
        "Go1Getup",
        "Go1Handstand",
        "Go1JoystickFlatTerrain",
        "Go1JoystickRoughTerrain",
        "H1InplaceGaitTracking",
        "H1JoystickGaitTracking",
        "Op3Joystick",
        "SpotFlatTerrainJoystick",
        "SpotGetup",
        "SpotJoystickGaitTracking",
        "T1JoystickFlatTerrain",
        "T1JoystickRoughTerrain",
        # Manipulation
        "AeroCubeRotateZAxis",
        "AlohaHandOver",
        "AlohaSinglePegInsertion",
        "LeapCubeReorient",
        "LeapCubeRotateZAxis",
        "PandaOpenCabinet",
        "PandaPickCube",
        "PandaPickCubeCartesian",
        "PandaPickCubeOrientation",
        "PandaRobotiqPushCube",
    ] = "CartpoleBalance"


class MuJoCoPlaygroundEnv(Env):
    def __init__(self, config: MuJoCoPlaygroundConfig, jit: bool = True):
        from mujoco_playground import registry

        self.env_cfg = registry.get_default_config(config.env_name)
        self._env = registry.load(config.env_name)

        self.obs_shape = (self._env.observation_size,)
        self.action_size = self._env.action_size

        # self.action_ranges = self._env.mj_model.actuator_ctrlrange
        # self.action_ranges = jnp.array(self.action_ranges)

        super().__init__(config, jit=jit)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        mjx_state = self._env.reset(key)

        return State(
            env_state=mjx_state,
            obs=jnp.array(mjx_state.obs),
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
            info={"step": jnp.int32(0)},
        )

    def _inner_step_fn(
        self, state: State, action: Array, key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        mjx_state = self._env.step(state.env_state, action)

        step = state.info["step"] + 1
        truncation = step >= self.env_cfg.episode_length
        done = mjx_state.done.astype(jnp.bool) | truncation

        step_output = StepOutput(
            reward=mjx_state.reward,
            done=done,
            truncation=truncation,
            info=mjx_state.info,
        )
        new_state = State(
            env_state=mjx_state,
            obs=jnp.array(mjx_state.obs),
            action_mask=jnp.ones(self.action_size, dtype=jnp.bool),
            info={"step": step},
        )

        return step_output, new_state

    def sample_action(self, key: Key[Array, ""], num_envs: int) -> Float[Array, "B A"]:
        """Sample random continuous actions in [-1, 1]."""
        return jax.random.uniform(key, (num_envs, self.action_size), minval=-1.0, maxval=1.0)

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        rendered = self._env.render([state.env_state], width=256, height=256)
        return np.array(rendered[0], dtype=np.uint8)

    def batch_render(self, states: StateWithMetrics) -> np.ndarray:
        num_envs = states.obs.shape[0]
        host_env_state = jax.device_get(states.env_state)
        env_states = [jax.tree.map(lambda x: x[i], host_env_state) for i in range(num_envs)]
        rendered = self._env.render(env_states, width=256, height=256)
        return np.array(rendered, dtype=np.uint8)
