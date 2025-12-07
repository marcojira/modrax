"""Gating module for GTrXL"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class Gating(nnx.Module):
    def __init__(self, features: int, bias_init: float = 0.0, *, rngs: nnx.Rngs):
        linear = partial(
            nnx.Linear,
            in_features=features,
            out_features=features,
            use_bias=False,
        )

        self.dense_r_y = linear(rngs=rngs)
        self.dense_r_x = linear(rngs=rngs)
        self.dense_z_y = linear(rngs=rngs)
        self.dense_z_x = linear(rngs=rngs)
        self.dense_h_y = linear(rngs=rngs)
        self.dense_h_rx = linear(rngs=rngs)

        self.gating_bias = nnx.Param(jnp.full((features,), bias_init))

    def __call__(self, x: Float[Array, "B ... D"], y: Float[Array, "B ... D"]):
        r = jax.nn.sigmoid(self.dense_r_x(x) + self.dense_r_y(y))
        z = jax.nn.sigmoid(self.dense_z_x(x) + self.dense_z_y(y) - self.gating_bias)
        h = jnp.tanh(self.dense_h_rx(r * x) + self.dense_h_y(y))
        g = (1 - z) * x + (z * h)
        return g
