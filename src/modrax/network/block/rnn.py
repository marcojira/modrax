"""Wrapper for Flax NNX recurrent networks (LSTM, GRU, SimpleRNN)."""

from typing import Any, Callable, Literal

from attr import dataclass
from flax import nnx
from jaxtyping import Array, Float

from modrax.network.block.base import BlockConfig, RecurrentBlock, RecurrentState


class RNNConfig(BlockConfig):
    cell_type: Literal["lstm", "gru", "simple"]
    gate_fn: Callable = nnx.sigmoid
    activation_fn: Callable = nnx.tanh
    unroll: int = 1
    optimized_lstm: bool = True  # Use optimized LSTM implementation (only for lstm cell_type)
    residual: bool = False  # Use residual connections (only for simple cell_type)


@dataclass
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

        self.cell = self._create_cell(input_shape, output_dim, rngs)
        self.rnn = nnx.RNN(
            cell=self.cell,
            return_carry=True,
            unroll=config.unroll,
        )

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

    def init_recurrent_state(self, batch_size: int) -> RNNRecurrentState:
        input_shape = (batch_size, self.cell.in_features)
        carry = self.cell.initialize_carry(input_shape)
        return RNNRecurrentState(carry=carry)

    def __call__(
        self,
        obs: Float[Array, "B T D"],
        recurrent_state: RNNRecurrentState,
    ) -> tuple[Float[Array, "B T H"], RNNRecurrentState]:
        """Forward pass for generation"""
        new_carry, out = self.rnn(obs, initial_carry=recurrent_state.carry)
        return out, RNNRecurrentState(carry=new_carry)  # type: ignore

    def train_forward(
        self,
        encoded_obs: Float[Array, "B T D"],
        init_recurrent_state: RNNRecurrentState,
    ) -> Float[Array, "B T H"]:
        """Forward pass for training (returns only output, not state)."""
        # TODO: Manual forward taking into account dones
        out, _ = self(encoded_obs, init_recurrent_state)
        return out
