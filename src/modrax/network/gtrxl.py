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

from modrax.network.gating import GatingTransformerLayer


@struct.dataclass
class GTrXLRecurrentState:
    memory: Float[Array, "B M L D"]
    mask: Bool[Array, "B 1 1 M+1"]


class GatedTransformerXL(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        num_layers: int,
        rollout_memory_len: int,
        segment_len: int,
        rngs: nnx.Rngs,
        gating: bool = False,
        gating_bias: float = 0.0,
    ):
        self.output_dim = input_dim
        self.qkv_features = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.gating = gating
        self.gating_bias = gating_bias
        self.segment_len = segment_len
        self.rollout_memory_len = rollout_memory_len

        @nnx.split_rngs(splits=self.num_layers)
        @nnx.vmap
        def make_transformer_layer(rng):
            return GatingTransformerLayer(
                num_heads=self.num_heads,
                io_features=self.qkv_features,
                qkv_features=self.qkv_features,
                gating=self.gating,
                gating_bias=self.gating_bias,
                rngs=rng,
            )

        self.tf_layers = make_transformer_layer(rngs)

    def initialize_carry(self, batch_size: int):
        memory = jnp.zeros(
            (batch_size, self.rollout_memory_len, self.num_layers, self.qkv_features)
        )
        mask = jnp.zeros((batch_size, 1, 1, 1 + self.rollout_memory_len))

        self.carry = nnx.Variable(GTrXLRecurrentState(memory=memory, mask=mask))

    def reset(self, done: Float[Array, " B"]):
        done = done[:, None, None, None]  # Add dimensions for broadcasting

        # Reset memory/mask for states that are done
        memory = jnp.where(done, 0, self.carry.memory)
        mask = jnp.where(done, 0, self.carry.mask)

        self.carry.value = GTrXLRecurrentState(memory=memory, mask=mask)

    def _forward(
        self,
        x: Float[Array, "B T D"],
        memory: Float[Array, "B M L D"],
        mask: Bool[Array, "B 1 1 M+1"],
    ):
        @nnx.scan(in_axes=(0, 2, nnx.Carry), out_axes=(nnx.Carry, 0))
        def layer_forward(layer, memory, x):
            out = layer(memory, x, mask=mask)

            return out, x

        y, hidden = layer_forward(self.tf_layers, memory, x)

        # Hidden: [L, B, T, D] -> [B, T, L, D]
        hidden = jnp.transpose(hidden, (1, 2, 0, 3))

        return hidden, y

    def train_forward(
        self,
        x: Float[Array, "B T D"],
        init_carry: GTrXLRecurrentState,
        saved_states: GTrXLRecurrentState,
    ):
        init_memory = init_carry.memory
        saved_memory = saved_states.memory
        mask = saved_states.mask

        # Split minibatch into segments (along time dimension)
        num_segments = x.shape[1] // self.segment_len
        segment_batch_size = x.shape[0] * num_segments
        segment_encoded_obs = jnp.reshape(
            x, (segment_batch_size, self.segment_len, *x.shape[2:])
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

        _, out = self._forward(segment_encoded_obs, memory, mask)
        return out.reshape((x.shape[0], x.shape[1], -1))

    def _eval_forward(self, x: Float[Array, "B D"]):
        # Reshape [B, D] -> [B, 1, D]
        x = x[:, None, :]
        memory, mask = self.carry.memory, self.carry.mask

        # Add attention to current observation
        mask = mask.at[:, :, :, -1].set(1)

        hidden, x = self._forward(x, memory, mask)

        # Memory is [B, M, L, D]. This adds the new memory to the end, like a queue
        # Roll memory and add new hidden state
        memory = jnp.roll(memory, -hidden.shape[1], axis=1)
        memory = memory.at[:, -hidden.shape[1] :].set(hidden)

        # Roll mask for next generation
        mask = jnp.roll(mask, -hidden.shape[1], axis=-1)
        new_carry = GTrXLRecurrentState(memory=memory, mask=mask)

        return new_carry, x

    def __call__(self, x: Float[Array, "B D"]):
        new_carry, x = self._eval_forward(x)

        # Create carry to store for trainining later
        store_carry = GTrXLRecurrentState(memory=new_carry.memory[:, -1], mask=self.carry.mask)

        self.carry.value = new_carry
        return store_carry, x[:, 0, :]
