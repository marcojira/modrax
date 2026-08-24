import dataclasses
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Key, Shaped


@dataclass(frozen=True)
class DiscreteActionSpec:
    num_actions: int


@dataclass(frozen=True)
class ContinuousActionSpec:
    shape: tuple[int, ...]
    low: Float[Array, "..."]
    high: Float[Array, "..."]


ActionSpec = DiscreteActionSpec | ContinuousActionSpec


@dataclass(frozen=True)
class EnvConfig:
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
    obs_shape: tuple[int, ...]
    action_spec: ActionSpec
    config: EnvConfig

    def __init__(self, config: EnvConfig):
        if config.optimistic_reset and not config.auto_reset:
            print("Optimistic resets require auto_reset=True. Proceeding with auto_resets")
            config = dataclasses.replace(config, auto_reset=True)

        self.config = config

    @property
    def action_size(self) -> int:
        """Return the discrete action count or continuous action dimension."""
        if isinstance(self.action_spec, DiscreteActionSpec):
            return self.action_spec.num_actions
        return math.prod(self.action_spec.shape)

    @nnx.jit(static_argnames=["self"])
    def reset(self, keys: Key[Array, " B"]) -> StateWithMetrics:
        """Reset B environments in parallel.

        Args:
            keys: B PRNG keys, one per environment.

        Returns:
            StateWithMetrics with episode_return and episode_length initialized to 0.
        """
        return jax.vmap(self._reset_single)(keys)

    @nnx.jit(static_argnames=["self"])
    def step(
        self,
        state: StateWithMetrics,
        action: Shaped[Array, "B ..."],
        keys: Key[Array, " B"],
    ) -> tuple[StepOutput, StateWithMetrics]:
        """Step B environments in parallel.

        Handles auto-reset and optimistic resets based on config.

        Args:
            state: Current batched environment state with metrics.
            action: Batched actions, one per environment.
            keys: B PRNG keys for stochastic transitions and resets.

        Returns:
            Tuple of (StepOutput, StateWithMetrics)
        """
        if self.config.optimistic_reset:
            num_resets = self.config.num_reset_envs
            reset_keys = jax.random.split(keys[0], num_resets)
            reset_state = jax.vmap(self._reset_single)(reset_keys)
            indices = jax.vmap(lambda k: jax.random.choice(k, jnp.arange(num_resets)))(keys)
            reset_state = jax.tree.map(lambda x: x[indices], reset_state)
        elif self.config.auto_reset:
            reset_state = jax.vmap(self._reset_single)(keys)
        else:
            reset_state = state

        return jax.vmap(self._step_single)(state, reset_state, action, keys)

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        """Render a single environment state to RGB image (H, W, 3) uint8."""
        raise NotImplementedError("Subclasses must implement render")

    def batch_render(self, states: StateWithMetrics) -> np.ndarray:
        """Render a batch of states [B ...], returning array of shape (B, H, W, 3) uint8."""
        frames = [
            self.render(jax.tree.map(lambda x: x[i], states)) for i in range(states.obs.shape[0])
        ]
        return np.stack(frames)

    def sample_action(self, key: Key[Array, ""], num_envs: int) -> Shaped[Array, "B ..."]:
        """Sample random actions for num_envs environments."""
        if isinstance(self.action_spec, DiscreteActionSpec):
            return jax.random.randint(key, (num_envs,), 0, self.action_spec.num_actions)

        return jax.random.uniform(
            key,
            (num_envs, *self.action_spec.shape),
            minval=self.action_spec.low,
            maxval=self.action_spec.high,
        )

    def _reset_single(self, key: Key[Array, ""]) -> StateWithMetrics:
        """Reset a single environment and wrap with metrics."""
        state = self._inner_reset_fn(key)
        return StateWithMetrics(
            env_state=state.env_state,
            obs=state.obs,
            action_mask=state.action_mask,
            info=state.info,
            episode_return=jnp.zeros(()),
            episode_length=jnp.zeros((), dtype=jnp.int32),
        )

    def _step_single(
        self,
        state: StateWithMetrics,
        reset_state: StateWithMetrics,
        action: Shaped[Array, "..."],
        key: Key[Array, ""],
    ) -> tuple[StepOutput, StateWithMetrics]:
        """Step a single environment with auto-reset and metric tracking."""
        inner_state = State(
            env_state=state.env_state,
            obs=state.obs,
            action_mask=state.action_mask,
            info=state.info,
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

        return step_output, StateWithMetrics(
            env_state=new_state.env_state,
            obs=new_state.obs,
            action_mask=new_state.action_mask,
            info=new_state.info,
            episode_return=episode_return,
            episode_length=episode_length,
        )

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        """Reset a single environment. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _inner_reset_fn")

    def _inner_step_fn(
        self, state: State, action: Shaped[Array, "..."], key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        """Step a single environment. Subclasses must override this."""
        raise NotImplementedError("Subclasses must implement _inner_step_fn")

    def __hash__(self) -> int:
        """Hash based on config."""
        return hash(str(dataclasses.asdict(self.config)))
