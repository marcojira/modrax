"""
Implementation of relative multi-head attention as described in
"Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

Follows the notation and description in "Stabilizing Transformers for Reinforcement Learning"

Inspired by:
- https://github.com/Reytuag/transformerXL_PPO_JAX
- https://github.com/kimiyoung/transformer-xl
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import LinearGeneral, default_kernel_init
from jaxtyping import Array, Float


class RelativeMultiHeadSelfAttention(nnx.Module):
    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int,
        out_features: int,
        dropout_rate: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        # Store args
        self.num_heads = num_heads
        self.in_features = in_features
        self.qkv_features = qkv_features
        self.out_features = out_features
        self.dropout_rate = dropout_rate

        self.head_dim = self.qkv_features // self.num_heads
        linear_general = partial(
            LinearGeneral,
            in_features=self.in_features,
            out_features=(self.num_heads, self.head_dim),
            kernel_init=default_kernel_init,  # Default from nnx
            bias_init=initializers.zeros_init(),  # Default from nnx
            use_bias=True,  # Default from nnx
            rngs=rngs,
        )

        # Layers
        self.query = linear_general(rngs=rngs)
        self.key = linear_general(rngs=rngs)
        self.value = linear_general(rngs=rngs)
        self.pos_map = linear_general(rngs=rngs)

        self.u = nnx.Param(jnp.zeros((self.num_heads, self.head_dim)))  # [H, D]
        self.v = nnx.Param(jnp.zeros((self.num_heads, self.head_dim)))  # [H, D]

        self.out = LinearGeneral(
            in_features=(self.num_heads, self.head_dim),
            out_features=self.out_features,
            kernel_init=default_kernel_init,  # Default from nnx
            bias_init=initializers.zeros_init(),  # Default from nnx
            use_bias=True,  # Default from nnx
            axis=(-2, -1),
            rngs=rngs,
        )

    def __call__(
        self,
        E: Float[Array, "B T H D"],
        E_tilde: Float[Array, "B M+T H D"],
        *,
        mask: Float[Array, "... T M+T"] | None = None,
    ):
        Q = self.query(E)  # [B, T, H, D]
        K, V = self.key(E_tilde), self.value(E_tilde)  # [B, M + T, H, D]

        # Embed position matrix
        pos_emb = make_pos_emb(E_tilde.shape[0], K.shape[1], E_tilde.shape[2])  # [B, M + T, F]
        R = self.pos_map(pos_emb)  # [B, M + T, H, D]

        # Attention
        out = dot_attention(Q, K, V, R, self.u, self.v, mask=mask)  # [B, T, H, D]
        return self.out(out)  # [B, T, F]


def make_pos_emb(batch_size: int, seq_length: int, embed_dim: int):
    positions = jnp.arange(seq_length - 1, -1, -1)[:, None]
    dim_indices = 2 * (jnp.arange(embed_dim)[None, :] // 2)  # [0, 0, 2, 2, 4, 4, ...]

    div_term = jnp.exp(dim_indices * (-jnp.log(10000.0) / embed_dim))
    angles = positions * div_term

    pos_emb = jnp.where(jnp.arange(embed_dim)[None, :] % 2 == 0, jnp.sin(angles), jnp.cos(angles))
    return jnp.repeat(pos_emb[None, :], repeats=batch_size, axis=0)


def rel_shift(x: Float[Array, "B H T T+M"], mask: bool = True):
    """
    x is [B, H, T, T + M]
    For each [B, H] we want to apply the following transform:

    (0,0)     ...  (0,T+M-1)
    ⋮               ⋮
    (T-1,0)   ...  (T-1,T+M-1)

    should become

    (0,T-1)   ...  (0,T+M-1)      0        ... 0     # Roll left by T
    (1,T-2)   ...  (1,T+M-1), (1,T+M-1), 0 ... 0     # Roll left by T-1
    ⋮               ⋮           ⋮             ⋮
    (T-1,0)   ...                        (T-1,T+M-1) # Roll left by 0

    i.e. there is a decreasing roll + masking
    """
    T, M = x.shape[2], x.shape[3] - x.shape[2]

    # Apply shift
    decreasing_shifts = -jnp.arange(T - 1, -1, -1)  # Shift by decreasing amount [T-1, ..., 0]
    # Creates a function that applies the correct shift to each high-dimensional row
    vmapped_shift = jax.vmap(
        lambda x, shift: jnp.roll(x, shift, axis=-1),  # Apply shift to columns of [B, H, T+M] row
        in_axes=(
            -2,  # Each row (axis position is -2)
            0,
        ),
        out_axes=-2,  # Put result back in position -2
    )
    x = vmapped_shift(x, decreasing_shifts)

    if mask:
        # Top right structure 0s have same structure as causal mask for M, T
        top_right_mask = make_causal_mask(M, T)
        x = x * top_right_mask

    return x


def dot_attention(
    Q: Float[Array, "B T H D"],
    K: Float[Array, "B M+T H D"],
    V: Float[Array, "B M+T H D"],
    R: Float[Array, "B M+T H D"],
    u: Float[Array, "H D"] | nnx.Param,
    v: Float[Array, "H D"] | nnx.Param,
    mask: Float[Array, "... T M+T"] | None = None,
):
    # Compute attention weights
    ab = jnp.einsum("...thd,...mhd->...htm", Q + u, K)  # [B, H, T, M + T]
    cd = jnp.einsum("...thd,...mhd->...htm", Q + v, R)  # [B, H, T, M + T]
    cd = rel_shift(cd)

    attn_weights = ab + cd  # [B, H, T, M + T]
    attn_weights = attn_weights / (jnp.sqrt(Q.shape[-1]))
    if mask is not None:
        attn_weights = jnp.where(mask, attn_weights, -jnp.inf)

    attn_weights = jax.nn.softmax(attn_weights)

    # Attention
    return jnp.einsum(
        "...htm,...mhd->...thd",
        attn_weights,
        V,
    )  # [B, T, H, D]


def make_causal_mask(M: int, T: int):
    """
    Makes a [1, 1, T, M + T] causal mask that ensures that the query does not attend
    to tokens after it. 1 when t
    """
    mask = jnp.tril(jnp.ones((T, T)))
    mask = jnp.concat([jnp.ones((T, M)), mask], axis=-1)
    return mask[None, None, ...]
