"""Simple implementation of MLP for general use"""

import math
from typing import Callable, Sequence

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from modrax.types import Shape


class MLP(nnx.Module):
    def __init__(
        self,
        input_shape: Shape | int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation_fn: Callable,
        rngs: nnx.Rngs,
        layer_norm: bool = False,
    ):
        self.input_dim = input_shape if isinstance(input_shape, int) else math.prod(input_shape)
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm

        # Build layers
        self.layers = []
        self.layer_norms = []
        layer_sizes = [self.input_dim] + list(self.hidden_dims)

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nnx.Linear(layer_sizes[i], layer_sizes[i + 1], rngs=rngs))
            if layer_norm:
                self.layer_norms.append(
                    nnx.LayerNorm(layer_sizes[i + 1], use_scale=False, use_bias=False, rngs=rngs)
                )

        # Output layer
        self.output_layer = nnx.Linear(layer_sizes[-1], output_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "B ..."]) -> Float[Array, "B ..."]:
        x = x.astype(jnp.float32)
        x = x.reshape(x.shape[0], -1)  # Flatten to [B, input_dim]

        # Forward pass through all hidden layers with activation
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation_fn(x)

        # Output layer without activation
        return self.output_layer(x)  # [B, output_dim]
