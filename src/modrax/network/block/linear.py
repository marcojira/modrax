"""Linear block for composable networks."""

import math

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from modrax.network.block.base import Block, BlockConfig
from modrax.types import Shape


class LinearConfig(BlockConfig):
    """Simple passthrough config as Linear has no configurable behavior."""

    pass


class Linear(Block):
    """Linear transformation block.

    Applies a single linear transformation (fully connected layer) to map
    from input_shape to output_dim. Input is automatically flattened.
    """

    def __init__(
        self,
        input_shape: Shape | int,
        output_dim: int,
        config: LinearConfig,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        self.output_dim = output_dim

        self.layer = nnx.Linear(self.input_dim, output_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "B ..."]) -> Float[Array, "B output_dim"]:
        x = x.astype(jnp.float32)
        x = x.reshape(x.shape[0], -1)  # Flatten to [B, input_dim]

        return self.layer(x)  # [B, output_dim]
