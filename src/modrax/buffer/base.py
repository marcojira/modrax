from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Int, Key


@struct.dataclass
class BufferState:
    """Immutable state for the replay buffer."""

    data: Any  # Pytree with arrays of shape (buffer_size, ...)
    priorities: Float[Array, " buffer_size"]
    position: Int[Array, ""]
    size: Int[Array, ""]


class ReplayBuffer:
    """Prioritized replay buffer for arbitrary pytree data."""

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.0,
        beta: float = 0.4,
        jit: bool = True,
    ):
        """Create a replay buffer.

        Args:
            max_size: Maximum buffer capacity.
            alpha: Priority exponent (0 = uniform, 1 = full prioritization).
            beta: Importance sampling exponent for bias correction.
            jit: Whether to JIT compile buffer operations.
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self._setup_fns(jit)

    def init(self, sample: Any) -> BufferState:
        """Initialize buffer state from a sample pytree."""
        data = jax.tree.map(
            lambda x: jnp.zeros((self.max_size, *x.shape[1:]), dtype=x.dtype),
            sample,
        )
        return BufferState(
            data=data,
            priorities=jnp.zeros(self.max_size),
            position=jnp.array(0, dtype=jnp.int32),
            size=jnp.array(0, dtype=jnp.int32),
        )

    def _add_fn(self, state: BufferState, data: Any) -> BufferState:
        """Add transitions with max priority."""
        num_samples = jax.tree.leaves(data)[0].shape[0]
        max_priority = jnp.maximum(state.priorities.max(), 1.0)
        priorities = jnp.full(num_samples, max_priority)

        # Compute indices for insertion (ring buffer)
        indices = (state.position + jnp.arange(num_samples)) % self.max_size

        # Update data and priorities
        new_data = jax.tree.map(lambda buf, new: buf.at[indices].set(new), state.data, data)
        new_priorities = state.priorities.at[indices].set(priorities)

        # Update position and size
        new_position = (state.position + num_samples) % self.max_size
        new_size = jnp.minimum(state.size + num_samples, self.max_size)

        return state.replace(
            data=new_data,
            priorities=new_priorities,
            position=new_position,
            size=new_size,
        )

    def _sample_fn(
        self,
        state: BufferState,
        key: Key[Array, ""],
        batch_size: int,
    ) -> tuple[Any, Int[Array, " batch_size"], Float[Array, " batch_size"]]:
        """Sample a batch using prioritized sampling."""
        # Compute sampling probabilities from priorities
        valid_priorities = jnp.where(
            jnp.arange(self.max_size) < state.size,
            state.priorities,
            0.0,
        )
        probs = valid_priorities**self.alpha
        probs = probs / probs.sum()

        # Sample indices based on priorities
        indices = jax.random.choice(key, self.max_size, shape=(batch_size,), p=probs, replace=True)

        # Compute importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        weights = (state.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Gather sampled data
        sampled_data = jax.tree.map(lambda x: x[indices], state.data)

        return sampled_data, indices, weights

    def _update_priorities_fn(
        self,
        state: BufferState,
        indices: Int[Array, " N"],
        priorities: Float[Array, " N"],
    ) -> BufferState:
        """Update priorities for sampled transitions."""
        new_priorities = state.priorities.at[indices].set(priorities)
        return state.replace(priorities=new_priorities)

    def _setup_fns(self, jit: bool):
        """Setup optionally JIT-compiled functions."""
        if jit:
            self._add = jax.jit(self._add_fn)
            self._sample = jax.jit(self._sample_fn, static_argnames=("batch_size",))
            self._update_priorities = jax.jit(self._update_priorities_fn)
        else:
            self._add = self._add_fn
            self._sample = self._sample_fn
            self._update_priorities = self._update_priorities_fn

    def add(self, state: BufferState, data: Any) -> BufferState:
        """Add transitions with max priority."""
        return self._add(state, data)

    def sample(
        self,
        state: BufferState,
        key: Key[Array, ""],
        batch_size: int,
    ) -> tuple[Any, Int[Array, " batch_size"], Float[Array, " batch_size"]]:
        """Sample a batch using prioritized sampling."""
        return self._sample(state, key, batch_size)

    def update_priorities(
        self,
        state: BufferState,
        indices: Int[Array, " N"],
        priorities: Float[Array, " N"],
    ) -> BufferState:
        """Update priorities for sampled transitions."""
        return self._update_priorities(state, indices, priorities)
