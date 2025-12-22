"""Network definition for network with recurrent trunk"""

from typing import Any

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from pydantic import ConfigDict

from modrax.network.block.base import Block, BlockConfig, RecurrentBlock, RecurrentState
from modrax.network.block.gtrxl import GTrXLConfig
from modrax.network.block.rnn import RNNConfig
from modrax.network.block_network import BlockNetwork, BlockNetworkConfig
from modrax.rollout.recurrent_rollout import RecurrentRolloutData
from modrax.types import Shape

RecurrentBlockConfig = RNNConfig | GTrXLConfig


class RecurrentNetworkConfig(BlockNetworkConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoders: dict[
        str, tuple[type[Block], BlockConfig, Shape | None]
    ]  # Mapping from input name to (BlockClass, BlockConfig, Shape). None = use obs_shape
    encoder_dim: int  # Output dimension for all encoders

    trunk: None = None
    trunk_dim: None = None

    recurrent: tuple[type[RecurrentBlock], BlockConfig]
    recurrent_dim: int

    heads: dict[
        str, tuple[type[Block], BlockConfig, int | None]
    ]  # Mapping output name to (BlockClass, BlockConfig, output_dim). None = use num_actions


class RecurrentNetwork(BlockNetwork):
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
        >>> from modrax.types import OBS_SHAPE, NUM_ACTIONS
        >>>
        >>> config = RecurrentNetworkConfig(
        ...     encoders={"obs": (MLP, MLPConfig(hidden_dims=(128,)), OBS_SHAPE)},
        ...     encoder_dim=128,
        ...     recurrent=(NnxRNN, RNNConfig(cell_type="lstm")),
        ...     recurrent_dim=256,
        ...     heads={
        ...         "policy": (MLP, MLPConfig(hidden_dims=(32,)), NUM_ACTIONS),
        ...         "value": (Linear, LinearConfig(), 1),
        ...     },
        ... )
        >>> network = RecurrentNetwork(
        ...     obs_shape=(4,),
        ...     num_actions=2,
        ...     config=config,
        ...     rngs=rngs,
        ... )
        >>> outputs, state = network({"obs": obs_batch}, recurrent_state)
        >>> # outputs = {"policy": [B, 2], "value": [B, 1]}
    """

    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        config: RecurrentNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.encoders = self.build_encoders(obs_shape, config, rngs)

        # Build recurrent block (takes concatenated encoders)
        recurrent_cls, recurrent_config = config.recurrent
        recurrent_input_dim = len(self.encoders) * config.encoder_dim
        self.recurrent_block = recurrent_cls(
            input_shape=recurrent_input_dim,
            output_dim=config.recurrent_dim,
            config=recurrent_config,
            rngs=rngs,
        )

        self.heads = self.build_heads(num_actions, config.recurrent_dim, config, rngs)

    def init_recurrent_state(self, *args, **kwargs) -> Any:
        return self.recurrent_block.init_recurrent_state(*args, **kwargs)

    def reset_recurrent_state(self, *args, **kwargs) -> Any:
        return self.recurrent_block.reset_recurrent_state(*args, **kwargs)

    def set_recurrent_state(self, *args, **kwargs) -> Any:
        self.recurrent_state = nnx.Cache(self.recurrent_block.init_recurrent_state(*args, **kwargs))

    def __call__(
        self,
        inputs: dict[str, Float[Array, "B ..."]],
        recurrent_state: RecurrentState | None = None,
    ) -> tuple[dict[str, Array], Any] | dict[str, Array]:
        """
        Forward pass for generation/rollout (single timestep).
        Supports being called with/without recurrent state.
        For the latter, the recurrent state is stored/updated in the network
        """
        if recurrent_state is None:
            use_recurrent_state = self.recurrent_state.value
        else:
            use_recurrent_state = recurrent_state

        # Encode each input
        encoded = []
        for name in sorted(self.encoders.keys()):
            x = self.encoders[name](inputs[name])  # [B, encoder_dim]
            encoded.append(x)

        # Concatenate encoders
        encoded = jnp.concatenate(encoded, axis=-1)  # [B, encoder_dim * num_inputs]

        x = encoded[:, None, :]  # Add time dimension [B, 1, D]
        x, output_recurrent_state = self.recurrent_block(x, use_recurrent_state)
        x = x.squeeze(1)  # Remove time dimension [B, D]

        # Generate outputs for each head
        outputs = {name: head(x) for name, head in self.heads.items()}

        if recurrent_state is None:
            self.recurrent_state.value = output_recurrent_state
            return outputs

        return outputs, output_recurrent_state

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
        x = self.recurrent_block.train_forward(x, data.recurrent_output, data.init_recurrent_state)

        # Flatten for heads, then unflatten results
        x = flatten(x)
        outputs = {name: unflatten(head(x)) for name, head in self.heads.items()}

        return outputs
