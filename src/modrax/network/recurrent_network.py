"""Network definition for network with recurrent trunk"""

from typing import Any

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from pydantic import ConfigDict

from modrax.network.base import Network, NetworkConfig
from modrax.network.block.base import Block, BlockConfig, RecurrentBlock, RecurrentState
from modrax.network.block.gtrxl import GTrXLConfig
from modrax.network.block.rnn import RNNConfig
from modrax.rollout.recurrent_rollout import RecurrentRolloutData
from modrax.types import Shape

RecurrentBlockConfig = RNNConfig | GTrXLConfig


class RecurrentNetworkConfig(NetworkConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoders: dict[
        str, tuple[type[Block], BlockConfig]
    ]  # Mapping from input name to (BlockClass, BlockConfig)
    encoder_dim: int  # Output dimension for all encoders

    recurrent: tuple[type[RecurrentBlock], BlockConfig]
    recurrent_dim: int

    heads: dict[
        str, tuple[type[Block], BlockConfig]
    ]  # Mapping output name to (BlockClass, BlockConfig)


class RecurrentNetwork(Network):
    """Block-based recurrent network with encoder → recurrent → heads structure.

    Always includes a recurrent block (RNN or GTrXL) between encoder and head blocks.
    Provides separate forward passes for generation/rollout and training.

    Architecture (e.g.)::

        input_1 ─► [Encoder 1] ─┐
                                ├─► concat ─► [Recurrent] ─┬─► [Head 1] ─► output_1
        input_2 ─► [Encoder 2] ─┘               ▲    │     └─► [Head 2] ─► output_2
                                                |    │
                                              state ─┘

    Example:
        >>> from modrax.network.block.linear import Linear, LinearConfig
        >>> from modrax.network.block.mlp import MLP, MLPConfig
        >>> from modrax.network.block.rnn import NnxRNN, RNNConfig
        >>> import jax.nn as jnn
        >>>
        >>> config = RecurrentNetworkConfig(
        ...     encoders={"obs": (Linear, LinearConfig())},
        ...     encoder_dim=128,
        ...     recurrent=(NnxRNN, RNNConfig(cell_type="lstm")),
        ...     recurrent_dim=256,
        ...     heads={
        ...         "policy": (MLP, MLPConfig(hidden_dims=(32,), activation_fn=jnn.relu)),
        ...         "value": (Linear, LinearConfig()),
        ...     },
        ... )
        >>> network = RecurrentNetwork(
        ...     input_shapes={"obs": (4,)},
        ...     output_dims={"policy": 4, "value": 1},
        ...     config=config,
        ...     rngs=rngs,
        ... )
        >>> outputs, state = network({"obs": obs_batch}, recurrent_state)
        >>> # outputs = {"policy": [B, 4], "value": [B, 1]}
    """

    def __init__(
        self,
        input_shapes: dict[str, Shape | int],  # Dict mapping input names to shapes
        output_dims: dict[str, int],  # Dict mapping head names to output dimensions
        config: RecurrentNetworkConfig,
        rngs: nnx.Rngs,
    ):
        # Build encoder block for each input
        self.encoders = {
            name: block_cls(
                input_shape=input_shapes[name],
                output_dim=config.encoder_dim,
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config) in config.encoders.items()
        }

        # Build recurrent block (takes concatenated encoders)
        recurrent_cls, recurrent_config = config.recurrent
        recurrent_input_dim = len(input_shapes) * config.encoder_dim
        self.recurrent_block = recurrent_cls(
            input_shape=recurrent_input_dim,
            output_dim=config.recurrent_dim,
            config=recurrent_config,
            rngs=rngs,
        )

        # Build head block for each output
        self.heads = {
            name: block_cls(
                input_shape=config.recurrent_dim,
                output_dim=output_dims[name],
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config) in config.heads.items()
        }

    def init_recurrent_state(self, *args, **kwargs) -> Any:
        return self.recurrent_block.init_recurrent_state(*args, **kwargs)

    def reset_recurrent_state(self, *args, **kwargs) -> Any:
        return self.recurrent_block.reset_recurrent_state(*args, **kwargs)

    def __call__(
        self,
        inputs: dict[str, Float[Array, "B ..."]],
        recurrent_state: RecurrentState,
    ) -> tuple[dict[str, Array], Any]:
        """
        Forward pass for generation/rollout (single timestep).
        Takes in previous recurrent tstate and updates recurrent state
        """
        # Encode each input
        encoded = []
        for name in sorted(self.encoders.keys()):
            x = self.encoders[name](inputs[name])  # [B, encoder_dim]
            encoded.append(x)

        # Concatenate encoders
        x = jnp.concatenate(encoded, axis=-1)  # [B, encoder_dim * num_inputs]
        x = x[:, None, :]  # Add time dimension [B, 1, D]
        x, recurrent_state = self.recurrent_block(x, recurrent_state)
        x = x.squeeze(1)  # Remove time dimension [B, D]

        # Generate outputs for each head
        outputs = {name: head(x) for name, head in self.heads.items()}

        return outputs, recurrent_state

    def train_forward(self, data: RecurrentRolloutData) -> dict[str, Array]:
        """Forward pass for training with sequential data."""
        batch_size, seq_len = data.trajectory.obs.shape[0], data.trajectory.obs.shape[1]

        def flatten(x):
            return jnp.reshape(x, (-1, *x.shape[2:]))

        def unflatten(x):
            return jnp.reshape(x, (batch_size, seq_len, *x.shape[1:]))

        # Encode each input: flatten [B, T, ...] to [B*T, ...], encode, then unflatten to [B, T, D]
        encoded = []
        for name in sorted(self.encoders.keys()):
            x = getattr(data.trajectory, name)
            x = self.encoders[name](flatten(x))  # [B*T, encoder_dim]
            x = unflatten(x)  # [B, T, encoder_dim]
            encoded.append(x)

        x = jnp.concatenate(encoded, axis=-1)  # [B, T, encoder_dim * num_inputs]

        # Recurrent processing
        x = self.recurrent_block.train_forward(x, data.recurrent_state, data.init_recurrent_state)

        # Flatten for heads, then unflatten results
        x = flatten(x)
        outputs = {name: unflatten(head(x)) for name, head in self.heads.items()}

        return outputs
