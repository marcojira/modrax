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

    def __init__(self, config: EnvConfig, jit: bool = True):
        if config.optimistic_reset and not config.auto_reset:
            print("Optimistic resets require auto_reset=True. Proceeding with auto_resets")
            config.auto_reset = True

        self.config = config
        maybe_jit = jax.jit if jit else lambda f: f

        def _reset_with_metrics(key: Key[Array, ""]) -> StateWithMetrics:
            state = self._inner_reset_fn(key)
            return StateWithMetrics(
                env_state=state.env_state,
                obs=state.obs,
                action_mask=state.action_mask,
                info=state.info,
                episode_return=jnp.zeros(()),
                episode_length=jnp.zeros((), dtype=jnp.int32),
            )

        def _step_with_metrics(
            state: StateWithMetrics,
            reset_state: StateWithMetrics,
            action: Float[Array, "..."],
            key: Key[Array, ""],
        ) -> tuple[StepOutput, StateWithMetrics]:
            inner_state = State(
                env_state=state.env_state,
                obs=state.obs,
                action_mask=state.action_mask,
                info=state.info,
            )
            step_output, new_state = self._inner_step_fn(inner_state, action, key)

            if config.auto_reset:
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
                episode_return = (state.episode_return + step_output.reward) * (
                    1 - step_output.done
                )
                episode_length = jnp.int32((state.episode_length + 1) * (1 - step_output.done))
            else:
                episode_return = state.episode_return + step_output.reward
                episode_length = state.episode_length + 1

            return step_output, StateWithMetrics(
                env_state=new_state.env_state,
                obs=new_state.obs,
                action_mask=new_state.action_mask,
                info=new_state.info,
                episode_return=episode_return,
                episode_length=episode_length,
            )

        vmapped_reset = jax.vmap(_reset_with_metrics)
        vmapped_step = jax.vmap(_step_with_metrics)

        def _batch_step(
            state: StateWithMetrics, action: Float[Array, "B ..."], keys: Key[Array, " B"]
        ) -> tuple[StepOutput, StateWithMetrics]:
            if config.optimistic_reset:
                num_resets = config.num_reset_envs
                reset_keys = jax.random.split(keys[0], num_resets)
                reset_state = jax.vmap(_reset_with_metrics)(reset_keys)
                indices = jax.vmap(lambda k: jax.random.choice(k, jnp.arange(num_resets)))(keys)
                reset_state = jax.tree.map(lambda x: x[indices], reset_state)
            else:
                reset_state = vmapped_reset(keys) if config.auto_reset else state
            return vmapped_step(state, reset_state, action, keys)

        self.reset = maybe_jit(vmapped_reset)
        self.step = maybe_jit(_batch_step)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        """Reset environment and return State. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _inner_reset_fn")

    def _inner_step_fn(
        self, state: State, action: Float[Array, "..."], key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        """Step environment and return (StepOutput, State). Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _inner_step_fn")

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        """
        Render a single environment state to RGB image.
        Returns RGB image np.array of shape (H, W, 3) with dtype uint8
        """
        raise NotImplementedError("Subclasses must implement render")

    def sample_action(self, key: Key[Array, ""], num_envs: int) -> Int[Array, " B"]:
        """Sample random actions for multiple environments.

        Returns actions with shape (num_envs,).
        """
        return jax.random.randint(key, (num_envs,), 0, self.action_size)

    def __hash__(self) -> int:
        """Hash based on config."""
        return hash(str(dataclasses.asdict(self.config)))
