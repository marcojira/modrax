import dataclasses
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int, Key

from modrax.types import Config, Shape


@dataclass
class EnvConfig(Config):
    env_name: str = ""
    auto_reset: bool = True
    optimistic_reset: bool = False
    num_reset_envs: int = 8  # number of resets to pre-compute per step


# Type alias for environment-specific state
EnvState = Any


class State(NamedTuple):
    env_state: EnvState
    obs: Float[Array, "..."]
    action_mask: Bool[Array, "..."]
    info: Any = None


class StateWithMetrics(NamedTuple):
    """Add episode-level metrics (return and length)."""

    env_state: EnvState
    obs: Float[Array, "..."]
    action_mask: Bool[Array, "..."]
    info: Any
    episode_return: Float[Array, ""]
    episode_length: Int[Array, ""]


class StepOutput(NamedTuple):
    reward: Float[Array, ""]
    done: Bool[Array, ""]
    truncation: Bool[Array, ""]
    info: Any


class Env:
    obs_shape: Shape
    action_size: int
    config: EnvConfig

    def __new__(cls, config: EnvConfig, jit: bool = True):
        """Factory method to create appropriate Env subclass based on config type."""
        # If called on base Env class, dispatch to appropriate subclass
        if cls is Env:
            from modrax.env.craftax import CraftaxConfig, CraftaxEnv
            from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
            from modrax.env.mujoco_playground import MuJoCoPlaygroundConfig, MuJoCoPlaygroundEnv
            from modrax.env.octax import OctaxConfig, OctaxEnv
            from modrax.env.pgx import PGXConfig, PGXEnv

            if isinstance(config, MuJoCoPlaygroundConfig):
                return MuJoCoPlaygroundEnv(config, jit)
            elif isinstance(config, PGXConfig):
                return PGXEnv(config, jit)
            elif isinstance(config, CraftaxConfig):
                return CraftaxEnv(config, jit)
            elif isinstance(config, OctaxConfig):
                return OctaxEnv(config, jit)
            elif isinstance(config, GymnaxConfig):
                return GymnaxEnv(config, jit)
            else:
                raise ValueError(f"Unknown env config type: {type(config)}")
        else:
            # If called on a subclass, use normal instantiation
            return super().__new__(cls)

    def __init__(self, config: EnvConfig, jit: bool = True):
        if config.optimistic_reset and not config.auto_reset:
            print("Optimistic resets require auto_reset=True. Proceeding with auto_resets")
            self.config.auto_reset = True

        self.config = config
        self._setup_fns(jit=jit)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        """Reset environment and return State. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _inner_reset_fn")

    def _inner_step_fn(
        self, state: State, action: Float[Array, "..."], key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        """Step environment and return (StepOutput, State). Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _inner_step_fn")

    def reset_fn(self, key: Key[Array, ""]) -> StateWithMetrics:
        """Reset environment with episode metrics initialization."""
        state = self._inner_reset_fn(key)
        return StateWithMetrics(
            env_state=state.env_state,
            obs=state.obs,
            action_mask=state.action_mask,
            info=state.info,
            episode_return=jax.numpy.zeros(()),
            episode_length=jax.numpy.zeros((), dtype=jax.numpy.int32),
        )

    def step_fn(
        self,
        state: StateWithMetrics,
        reset_state: StateWithMetrics,
        action: Float[Array, "..."],
        key: Key[Array, ""],
    ) -> tuple[StepOutput, StateWithMetrics]:
        """Step environment with return/length accumulation and optional auto-reset."""
        inner_state = State(
            env_state=state.env_state, obs=state.obs, action_mask=state.action_mask, info=state.info
        )
        step_output, new_state = self._inner_step_fn(inner_state, action, key)

        if self.config.auto_reset:
            new_state = jax.lax.cond(
                step_output.done > 0,
                lambda: State(
                    env_state=reset_state.env_state,
                    obs=reset_state.obs,
                    action_mask=reset_state.action_mask,
                    info=reset_state.info,
                ),
                lambda: new_state,
            )
            episode_return = (state.episode_return + step_output.reward) * (1 - step_output.done)
            episode_length = jnp.int32((state.episode_length + 1) * (1 - step_output.done))
        else:
            episode_return = state.episode_return + step_output.reward
            episode_length = state.episode_length + 1

        new_state_with_metrics = StateWithMetrics(
            env_state=new_state.env_state,
            obs=new_state.obs,
            action_mask=new_state.action_mask,
            info=new_state.info,
            episode_return=episode_return,
            episode_length=episode_length,
        )
        return step_output, new_state_with_metrics

    def _setup_fns(self, jit: bool):
        """Setup vmapped and optionally JIT-compiled functions."""
        reset_fn = jax.vmap(self.reset_fn)
        step_fn = jax.vmap(self.step_fn)

        def batch_step(
            state: StateWithMetrics, action: Float[Array, "B ..."], keys: Key[Array, " B"]
        ) -> tuple[StepOutput, StateWithMetrics]:
            if self.config.optimistic_reset:
                num_resets = self.config.num_reset_envs
                reset_keys = jax.random.split(keys[0], num_resets)
                reset_state = jax.vmap(self.reset_fn)(reset_keys)

                # Assign a reset state to each environment
                indices = jax.vmap(lambda k: jax.random.choice(k, jnp.arange(num_resets)))(keys)
                reset_state = jax.tree.map(lambda x: x[indices], reset_state)
            else:
                reset_state = self._reset_fn(keys) if self.config.auto_reset else state
            return step_fn(state, reset_state, action, keys)

        if jit:
            reset_fn = jax.jit(reset_fn)
            batch_step = jax.jit(batch_step)

        self._reset_fn = reset_fn
        self._step_fn = batch_step

    def reset(self, keys: Key[Array, " B"]) -> StateWithMetrics:
        """Reset environments with given keys."""
        return self._reset_fn(keys)

    def step(
        self, state: StateWithMetrics, action: Float[Array, "B ..."], keys: Key[Array, " B"]
    ) -> tuple[StepOutput, StateWithMetrics]:
        """Step environments with given state, actions, and keys."""
        return self._step_fn(state, action, keys)

    def sample_action(self, key: Key[Array, ""], num_envs: int) -> Int[Array, " B"]:
        """Sample random actions for multiple environments.

        Returns actions with shape (num_envs,).
        """
        return jax.random.randint(key, (num_envs,), 0, self.action_size)

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        """
        Render a single environment state to RGB image.
        Returns RGB image np.array of shape (H, W, 3) with dtype uint8
        """
        raise NotImplementedError("Subclasses must implement render")

    def __hash__(self) -> int:
        """Hash based on config."""
        return hash(str(dataclasses.asdict(self.config)))
