"""MLP block for composable networks."""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from pydantic import ConfigDict

from modrax.network.block.base import Block, BlockConfig
from modrax.network.module.mlp import MLP as MLPModule
from modrax.types import Shape


class MLPConfig(BlockConfig):
    """Configuration for MLP block.

    Attributes:
        hidden_dims: Sequence of hidden layer dimensions
        activation_fn: Activation function to apply after each hidden layer
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_dims: Sequence[int]
    activation_fn: Callable = jax.nn.relu


class MLP(Block):
    """Wrapper around the MLP module that follows the Block API."""

    def __init__(
        self,
        input_shape: Shape | int,
        output_dim: int,
        config: MLPConfig,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLPModule(
            input_shape=input_shape,
            hidden_dims=config.hidden_dims,
            output_dim=output_dim,
            activation_fn=config.activation_fn,
            rngs=rngs,
        )

    def __call__(self, x: Float[Array, "B ..."]) -> Float[Array, "B output_dim"]:
        """Forward pass through MLP (the module handles the flattening)"""
        return self.mlp(x)
