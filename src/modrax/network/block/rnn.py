"""Wrapper for Flax NNX recurrent networks (LSTM, GRU, SimpleRNN)."""

from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Bool, Float

from modrax.network.block.base import BlockConfig, RecurrentBlock, RecurrentState


class RNNConfig(BlockConfig):
    cell_type: Literal["lstm", "gru", "simple"]
    gate_fn: Callable = nnx.sigmoid
    activation_fn: Callable = nnx.tanh
    unroll: int = 1
    optimized_lstm: bool = True  # Use optimized LSTM implementation (only for lstm cell_type)
    residual: bool = False  # Use residual connections (only for simple cell_type)
    num_layers: int = 1


@struct.dataclass
class RNNRecurrentState(RecurrentState):
    carry: Any  # Cell-specific carry (tuple for LSTM, array for GRU/Simple)


class NnxRNN(RecurrentBlock):
    def __init__(
        self,
        input_shape: int,
        output_dim: int,
        config: RNNConfig,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.output_dim = output_dim
        self.cell_type = config.cell_type
        self.gate_fn = config.gate_fn
        self.activation_fn = config.activation_fn
        self.unroll = config.unroll
        self.optimized_lstm = config.optimized_lstm
        self.residual = config.residual

        self.cells = [self._create_cell(input_shape, output_dim, rngs)] + [
            self._create_cell(output_dim, output_dim, rngs) for _ in range(config.num_layers - 1)
        ]
        self.rnns = [
            nnx.RNN(cell=cell, return_carry=True, unroll=config.unroll) for cell in self.cells
        ]

    def _create_cell(self, in_features: int, output_dim: int, rngs: nnx.Rngs):
        if self.cell_type == "lstm":
            cell_class = nnx.OptimizedLSTMCell if self.optimized_lstm else nnx.LSTMCell
            return cell_class(
                in_features=in_features,
                hidden_features=output_dim,
                gate_fn=self.gate_fn,
                activation_fn=self.activation_fn,
                rngs=rngs,
            )
        elif self.cell_type == "gru":
            return nnx.GRUCell(
                in_features=in_features,
                hidden_features=output_dim,
                gate_fn=self.gate_fn,
                activation_fn=self.activation_fn,
                rngs=rngs,
            )
        elif self.cell_type == "simple":
            return nnx.SimpleCell(
                in_features=in_features,
                hidden_features=output_dim,
                activation_fn=self.activation_fn,
                residual=self.residual,
                rngs=rngs,
            )
        else:
            raise ValueError(f"Unknown cell_type: {self.cell_type}")

    def stack_carrys(self, carrys):
        def stack(x):
            return jnp.stack(x, axis=1)

        if self.cell_type == "lstm":
            carry = (
                stack([x[0] for x in carrys]),
                stack([x[1] for x in carrys]),
            )
        else:
            carry = stack(carrys)

        return carry

    def init_recurrent_state(self, batch_size: int) -> RNNRecurrentState:
        """Creates carry of shape [B, L, D]"""
        carrys = [cell.initialize_carry((batch_size, cell.in_features)) for cell in self.cells]

        return RNNRecurrentState(carry=self.stack_carrys(carrys))

    def reset_recurrent_state(
        self, recurrent_state: RNNRecurrentState, done: Bool[Array, " B"]
    ) -> RNNRecurrentState:
        """Reset carry to zeros for episodes that are done."""

        def reset_carry(carry):
            return jnp.where(done[:, None, None], 0, carry)

        new_carry = jax.tree.map(reset_carry, recurrent_state.carry)
        return RNNRecurrentState(carry=new_carry)

    def __call__(
        self,
        obs: Float[Array, "B T D"],
        recurrent_state: RNNRecurrentState,
    ) -> tuple[Float[Array, "B T H"], RNNRecurrentState]:
        """Forward pass for generation"""
        prev_carry = recurrent_state.carry

        x, carrys = obs, []
        for i, layer in enumerate(self.rnns):
            new_carry, x = layer(x, initial_carry=jax.tree.map(lambda x: x[:, i], prev_carry))
            carrys.append(new_carry)

        return x, RNNRecurrentState(carry=self.stack_carrys(carrys))  # type: ignore

    def train_forward(
        self,
        encoded_obs: Float[Array, "B T D"],
        dones: Float[Array, "B T"],
        init_recurrent_state: RNNRecurrentState,
    ) -> Float[Array, "B T H"]:
        """Forward pass for training (returns only output, not state)."""

        def step(carry, obs_done):
            network, recurrent_state = carry
            obs, done = obs_done  # obs: [B, D], done: [B]
            recurrent_state = network.reset_recurrent_state(recurrent_state, done)
            x, recurrent_state = network(obs[:, None, :], recurrent_state)
            return (network, recurrent_state), x[:, 0]

        # Transpose to [T, B, ...] for scan over time
        obs_t = jnp.transpose(encoded_obs, (1, 0, 2))
        dones_t = jnp.transpose(dones, (1, 0))

        _, out_t = nnx.scan(step)((self, init_recurrent_state), (obs_t, dones_t))

        return jnp.transpose(out_t, (1, 0, 2))
