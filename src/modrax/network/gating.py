"""
Gating module for GTrXL

Inspired by:
- https://github.com/Reytuag/transformerXL_PPO_JAX
- https://github.com/kimiyoung/transformer-xl
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from modrax.network.relative_attention import RelativeMultiHeadSelfAttention


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


class GatingTransformerLayer(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        io_features: int,
        qkv_features: int,
        gating: bool = False,
        gating_bias: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.io_features = io_features
        self.qkv_features = qkv_features
        self.gating = gating
        self.gating_bias = gating_bias

        self.attention = RelativeMultiHeadSelfAttention(
            num_heads=num_heads,
            in_features=io_features,
            qkv_features=qkv_features,
            out_features=io_features,
            rngs=rngs,
        )

        self.ln1 = nnx.LayerNorm(io_features, rngs=rngs)
        self.ln2 = nnx.LayerNorm(io_features, rngs=rngs)

        self.dense1 = nnx.Linear(io_features, io_features, rngs=rngs)
        self.dense2 = nnx.Linear(io_features, io_features, rngs=rngs)

        if gating:
            self.gate1 = Gating(io_features, gating_bias, rngs=rngs)
            self.gate2 = Gating(io_features, gating_bias, rngs=rngs)

    def __call__(
        self,
        M: Float[Array, "B M H D"],
        E: Float[Array, "B T H D"],
        mask: Float[Array, "... T M+T"] | None,
    ):
        E_tilde = jnp.concat([M, E], axis=1)  # [B, M+T, H, D]
        E_tilde = self.ln1(E_tilde)
        E = self.ln1(E)  # Not in paper but done by https://github.com/Reytuag/transformerXL_PPO_JAX

        Y_bar = self.attention(E, E_tilde, mask=mask)

        if self.gating:
            Y = self.gate1(E, jax.nn.relu(Y_bar))
        else:
            Y = E + Y_bar

        E_bar = self.dense1(self.ln2(Y))
        E_bar = self.dense2(jax.nn.gelu(E_bar))

        if self.gating:
            # https://github.com/Reytuag/transformerXL_PPO_JAX version
            # E = self.gate2(E_bar, jax.nn.relu(Y))

            E = self.gate2(Y, jax.nn.relu(E_bar))  # Paper version
        else:
            E = E_bar + Y

        return E  # [B, T, D]
