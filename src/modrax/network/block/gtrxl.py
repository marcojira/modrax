"""
Implementation of GTrXL as described in "Stabilizing Transformers for Reinforcement Learning"
Follows the notation and description in "Stabilizing Transformers for Reinforcement Learning"

Inspired by:
- https://github.com/Reytuag/transformerXL_PPO_JAX
- https://github.com/kimiyoung/transformer-xl
"""

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Bool, Float

from modrax.network.block.base import BlockConfig, RecurrentBlock, RecurrentState
from modrax.network.module.gating import Gating
from modrax.network.module.relative_attention import RelativeMultiHeadSelfAttention


class GTrXLConfig(BlockConfig):
    num_heads: int
    num_layers: int
    rollout_memory_len: int  # Memory length during rollout
    segment_len: int  # Length of segments during training
    gating: bool = False
    gating_bias: float = 0.0


@struct.dataclass
class GTrXLRecurrentState(RecurrentState):
    memory: Float[Array, "B M L D"]
    mask: Bool[Array, "B 1 1 M+1"]


class GatedTransformerXL(RecurrentBlock):
    def __init__(
        self,
        input_shape: int,
        output_dim: int,
        config: GTrXLConfig,
        rngs: nnx.Rngs,
    ):
        # For GTrXL, input and output dimensions must be equal
        assert input_shape == output_dim, "GTrXL requires input_shape == output_dim"

        self.config = config
        self.output_dim = output_dim
        self.qkv_features = input_shape
        self.num_heads = config.num_heads
        self.num_layers = config.num_layers
        self.gating = config.gating
        self.gating_bias = config.gating_bias
        self.segment_len = config.segment_len
        self.rollout_memory_len = config.rollout_memory_len

        @nnx.split_rngs(splits=self.num_layers)
        @nnx.vmap
        def make_transformer_layer(rng):
            return TransformerLayer(
                num_heads=self.num_heads,
                io_features=self.qkv_features,
                qkv_features=self.qkv_features,
                gating=self.gating,
                gating_bias=self.gating_bias,
                rngs=rng,
            )

        self.tf_layers = make_transformer_layer(rngs)

    def _forward(
        self,
        obs: Float[Array, "B T D"],
        memory: Float[Array, "B M L D"],
        mask: Bool[Array, "B 1 1 M+1"],
    ):
        @nnx.scan(in_axes=(0, 2, nnx.Carry), out_axes=(nnx.Carry, 0))
        def layer_forward(layer, memory, x):
            out = layer(memory, x, mask=mask)

            return out, x

        y, hidden = layer_forward(self.tf_layers, memory, obs)

        # Hidden: [L, B, T, D] -> [B, T, L, D]
        hidden = jnp.transpose(hidden, (1, 2, 0, 3))

        return y, hidden

    def init_recurrent_state(self, batch_size: int):
        init_memory = jnp.zeros(
            (
                batch_size,
                self.rollout_memory_len,
                self.num_layers,
                self.qkv_features,
            )
        )
        init_mask = jnp.zeros((batch_size, 1, 1, 1 + self.rollout_memory_len))

        return GTrXLRecurrentState(memory=init_memory, mask=init_mask)

    def reset_recurrent_state(self, recurrent_state: GTrXLRecurrentState, done: Bool[Array, " B"]):
        done = done[:, None, None, None]  # Add dimensions for broadcasting

        # Reset memory/mask for states that are done
        memory = jnp.where(done, 0, recurrent_state.memory)
        mask = jnp.where(done, 0, recurrent_state.mask)

        return GTrXLRecurrentState(memory=memory, mask=mask)

    def __call__(
        self,
        encoded_obs: Float[Array, "B T D"],
        recurrent_state: GTrXLRecurrentState,
    ):
        """Forward pass for generation/rollout with memory queue update."""
        memory, mask = recurrent_state.memory, recurrent_state.mask

        # Add attention to current observation
        mask = mask.at[:, :, :, -1].set(1)

        y, hidden = self._forward(encoded_obs, memory, mask)

        # Memory is B x M x L x D. This adds the new memory to the end, like a queue
        # Roll memory and add new hidden state
        memory = jnp.roll(memory, -hidden.shape[1], axis=1)
        memory = memory.at[:, -hidden.shape[1] :].set(hidden)

        # Roll mask for next generation
        mask = jnp.roll(mask, -hidden.shape[1], axis=-1)

        return y, GTrXLRecurrentState(memory=memory, mask=mask)

    def train_forward(
        self,
        encoded_obs: Float[Array, "B T D"],
        recurrent_state: GTrXLRecurrentState,
        init_recurrent_state: GTrXLRecurrentState,
    ):
        saved_memory = recurrent_state.memory
        init_memory = init_recurrent_state.memory
        mask = recurrent_state.mask

        # Split minibatch into segments (along time dimension)
        num_segments = encoded_obs.shape[1] // self.segment_len
        segment_batch_size = encoded_obs.shape[0] * num_segments
        segment_encoded_obs = jnp.reshape(
            encoded_obs, (segment_batch_size, self.segment_len, *encoded_obs.shape[2:])
        )  # [B * NumSeg, SegLen, D]

        """ MEMORY """
        memory = jnp.reshape(
            saved_memory,
            (saved_memory.shape[0], num_segments, self.segment_len, *saved_memory.shape[2:]),
        )  # [B, NumSeg, SegLen, L, D]

        def update_memory(prev_memory, curr_memory):
            new_memory = jnp.concat([prev_memory[:, self.segment_len :], curr_memory], axis=1)
            return new_memory, prev_memory

        # Scan over segments
        _, segment_memory = nnx.scan(
            update_memory,
            in_axes=(nnx.Carry, 1),
            out_axes=(nnx.Carry, 1),
        )(init_memory, memory)  # [B, NumSeg, M, L, D]

        memory = jnp.reshape(
            segment_memory, (segment_batch_size, init_memory.shape[1], *segment_memory.shape[3:])
        )  # [B * NumSeg, M, L, D]

        """ MASK """
        mask = mask.at[:, :, :, -1].set(1)  # Add attention to the current observation

        # Reshape per segment
        mask = jnp.reshape(
            mask, (segment_batch_size, 1, self.segment_len, mask.shape[-1])
        )  # [B * NumSeg, 1, SegLen, M+1]
        mask = jnp.concat(
            [mask, jnp.zeros((segment_batch_size, 1, self.segment_len, self.segment_len - 1))],
            axis=-1,
        )  # [B * NumSeg, 1, SegLen, M+SegLen]

        # Varying roll so that each segment element only attends to last M + itself
        mask = jax.vmap(lambda x, shift: jnp.roll(x, shift, -1), in_axes=(-2, 0), out_axes=-2)(
            mask, jnp.arange(self.segment_len)
        )

        out, _ = self._forward(segment_encoded_obs, memory, mask)
        return out


class TransformerLayer(nnx.Module):
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
