from typing import Any

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float, Int, Key


@struct.dataclass
class BufferState:
    data: Any  # Pytree with arrays of shape (buffer_size, ...)
    priorities: Float[Array, " buffer_size"]
    position: Int[Array, ""]
    size: Int[Array, ""]


class ReplayBuffer:
    """Prioritized replay buffer for arbitrary pytree data."""

    def __init__(
        self,
        max_size: int,
        alpha: float = 0.0,  # Priority exponent (0 = uniform, 1 = full prioritization).
        beta: float = 0.4,  # Importance sampling exponent for bias correction.
    ):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta

        # Public API
        self.add = jax.jit(self._add)
        self.sample = jax.jit(self._sample, static_argnames=("batch_size",))
        self.update_priorities = jax.jit(self._update_priorities)

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

    def _add(self, state: BufferState, data: Any) -> BufferState:
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

        return BufferState(
            data=new_data,
            priorities=new_priorities,
            position=new_position,
            size=new_size,
        )

    def _sample(
        self, state: BufferState, key: Key[Array, ""], batch_size: int
    ) -> tuple[Any, Int[Array, " batch_size"], Float[Array, " batch_size"]]:
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

    def _update_priorities(
        self, state: BufferState, indices: Int[Array, " N"], priorities: Float[Array, " N"]
    ) -> BufferState:
        new_priorities = state.priorities.at[indices].set(priorities)
        return state.replace(priorities=new_priorities)  # type: ignore
