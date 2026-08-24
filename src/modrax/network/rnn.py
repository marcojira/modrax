"""Wrapper for Flax NNX recurrent networks (LSTM, GRU, SimpleRNN)."""

from typing import Callable, Literal

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


class NnxRNN(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rngs: nnx.Rngs,
        cell_type: Literal["lstm", "gru", "simple"] = "lstm",
        gate_fn: Callable = nnx.sigmoid,
        activation_fn: Callable = nnx.tanh,
        optimized_lstm: bool = True,
        residual: bool = False,
        num_layers: int = 1,
    ):
        self.output_dim = output_dim
        self.cell_type = cell_type
        self.gate_fn = gate_fn
        self.activation_fn = activation_fn
        self.optimized_lstm = optimized_lstm
        self.residual = residual
        self.rngs = rngs

        # TODO: VMAP this
        self.cells = nnx.List(
            [self._create_cell(input_dim, output_dim, rngs)]
            + [self._create_cell(output_dim, output_dim, rngs) for _ in range(num_layers - 1)]
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

    def initialize_carry(self, batch_size: int):
        self.carry = nnx.Variable(
            [
                cell.initialize_carry((batch_size, cell.in_features), rngs=self.rngs)
                for cell in self.cells
            ]
        )

    def _reset_carry(self, carry, done):
        return jax.tree.map(lambda c: jnp.where(done[:, None], 0, c), carry)

    def reset(self):
        self.carry.set_value(jax.tree.map(jnp.zeros_like, self.carry.get_value()))

    def reset_episodes(self, done: Array):
        self.carry.set_value(self._reset_carry(self.carry.get_value(), done))

    def _step(self, carry, x: Float[Array, "B D"]):
        """Single forward step through all RNN layers."""
        new_carry = []
        for i, layer in enumerate(self.cells):
            curr_carry, x = layer(carry[i], x)
            new_carry.append(curr_carry)
        return new_carry, x

    def train_forward(
        self, x: Float[Array, "B T D"], dones: Float[Array, "B T"], init_carry
    ) -> Float[Array, "B T H"]:
        """Scan through time dimension, resetting carry at episode boundaries."""

        def step(carry, inputs):
            x, done = inputs
            new_carry, x = self._step(carry, x)
            new_carry = self._reset_carry(new_carry, done)
            return new_carry, x

        _, out = nnx.scan(step, in_axes=(nnx.Carry, 1), out_axes=(nnx.Carry, 1))(
            init_carry, (x, dones)
        )
        return out

    def _eval_forward(self, x: Float[Array, "B D"]):
        """Single step without advancing stored carry."""
        carry, out = self._step(self.carry.get_value(), x)
        return carry, out

    def __call__(self, x: Float[Array, "B D"]):
        """Single eval step, advances stored carry. Returns (store_carry, out)."""
        new_carry, out = self._step(self.carry.get_value(), x)
        self.carry.set_value(new_carry)
        return new_carry, out
