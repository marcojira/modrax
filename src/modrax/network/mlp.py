"""Simple implementation of MLP for general use"""

from typing import Callable, Sequence

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class MLP(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation_fn: Callable,
        rngs: nnx.Rngs,
        layer_norm: bool = False,
    ):
        self.activation_fn = activation_fn
        self.layer_norm = layer_norm

        # Build layers
        self.layers = []
        self.layer_norms = []
        layer_sizes = [input_dim] + list(hidden_dims)

        for i in range(len(layer_sizes) - 1):
            self.layers.append(nnx.Linear(layer_sizes[i], layer_sizes[i + 1], rngs=rngs))
            if layer_norm:
                self.layer_norms.append(
                    nnx.LayerNorm(layer_sizes[i + 1], use_scale=False, use_bias=False, rngs=rngs)
                )

        # Output layer
        self.output_layer = nnx.Linear(layer_sizes[-1], output_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        """Apply MLP to the last dimension, broadcasting over all leading dims."""
        x = x.astype(jnp.float32)

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.layer_norm:
                x = self.layer_norms[i](x)
            x = self.activation_fn(x)

        return self.output_layer(x)
