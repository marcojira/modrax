"""Running statistics normalization block."""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from modrax.network.block.base import Block, BlockConfig
from modrax.types import Shape


class RunningNormConfig(BlockConfig):
    epsilon: float = 1e-8


class RunningNorm(Block):
    """Normalizes inputs using running mean and variance (Welford's algorithm)."""

    def __init__(
        self,
        input_shape: Shape | int,
        output_dim: int,
        config: RunningNormConfig,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        self.output_dim = self.input_dim
        self.epsilon = config.epsilon

        self.count = nnx.BatchStat(jnp.zeros(()))
        self.mean = nnx.BatchStat(jnp.zeros((self.input_dim,)))
        self.var = nnx.BatchStat(jnp.ones((self.input_dim,)))

    def update(self, x: Float[Array, "B ..."]):
        """Update running statistics with a new batch."""
        x = x.reshape(-1, self.input_dim)
        batch_count = x.shape[0]
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0)

        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + delta**2 * self.count * batch_count / total_count
        )

        self.count.value = total_count
        self.mean.value = new_mean
        self.var.value = m2 / total_count

    def __call__(self, x: Float[Array, "B ..."]) -> Float[Array, "B D"]:
        x = x.reshape(x.shape[0], -1)
        return (x - self.mean) / jnp.sqrt(self.var + self.epsilon)
