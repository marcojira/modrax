"""Network definition for standard block-based feedforward network."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from pydantic import ConfigDict

from modrax.network.base import Network, NetworkConfig
from modrax.network.block.base import Block, BlockConfig
from modrax.rollout.base import RolloutData
from modrax.types import Shape


class BlockNetworkConfig(NetworkConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoders: dict[
        str, tuple[type[Block], BlockConfig, Shape | None]
    ]  # Mapping from input name to (BlockClass, BlockConfig, Shape). OBS_SHAPE/None = use obs_shape
    encoder_dim: int  # Output dimension for all encoders

    trunk: tuple[type[Block], BlockConfig]
    trunk_dim: int  # Output dimension for trunk

    heads: dict[
        str, tuple[type[Block], BlockConfig, int | None]
    ]  # Mapping output name to (BlockClass, BlockConfig, output_dim). NUM_ACTIONS/None = use num_actions


class BlockNetwork(Network):
    """Block-based standard feedforward network with configurable encoders, trunk, and heads.

    Takes a dict of inputs, encodes each through separate encoder blocks,
    concatenates them, passes through a shared trunk block, then produces outputs
    via separate head blocks.

    Architecture (e.g.)::

        input_1 ─► [Encoder 1] ─┐
                                ├─► concat ─► [Trunk] ─┬─► [Head 1] ─► output_1
        input_2 ─► [Encoder 2] ─┘                      └─► [Head 2] ─► output_2

    Example:
        >>> from modrax.network.block.linear import Linear, LinearConfig
        >>> from modrax.network.block.mlp import MLP, MLPConfig
        >>> from modrax.types import OBS_SHAPE, NUM_ACTIONS
        >>> import jax.nn as jnn
        >>>
        >>> config = BlockNetworkConfig(
        ...     encoders={"obs": (MLP, MLPConfig(hidden_dims=(32,)), OBS_SHAPE)},
        ...     encoder_dim=32,
        ...     trunk=(MLP, MLPConfig(hidden_dims=(64, 64))),
        ...     trunk_dim=64,
        ...     heads={
        ...         "policy": (MLP, MLPConfig(hidden_dims=(32,)), NUM_ACTIONS),
        ...         "value": (Linear, LinearConfig(), 1),
        ...     },
        ... )
        >>> network = BlockNetwork(
        ...     obs_shape=(4,),
        ...     num_actions=2,
        ...     config=config,
        ...     rngs=rngs,
        ... )
        >>> outputs = network({"obs": obs_batch})
        >>> # outputs = {"policy": [B, 2], "value": [B, 1]}
    """

    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        config: BlockNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.encoders = self.build_encoders(obs_shape, config, rngs)

        # Build trunk block (takes concatenated encoders)
        trunk_cls, trunk_config = config.trunk
        trunk_input_dim = len(self.encoders) * config.encoder_dim
        self.trunk = trunk_cls(
            input_shape=trunk_input_dim,
            output_dim=config.trunk_dim,
            config=trunk_config,
            rngs=rngs,
        )

        self.heads = self.build_heads(num_actions, config.trunk_dim, config, rngs)

    def build_encoders(
        self,
        obs_shape: Shape,
        config: BlockNetworkConfig,
        rngs: nnx.Rngs,
    ) -> dict[str, Block]:
        """Build encoder blocks for each input."""
        return {
            name: block_cls(
                input_shape=obs_shape if shape is None else shape,
                output_dim=config.encoder_dim,
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config, shape) in config.encoders.items()
        }

    def build_heads(
        self,
        num_actions: int,
        feature_dim: int,
        config: BlockNetworkConfig,
        rngs: nnx.Rngs,
    ) -> dict[str, Block]:
        """Build head blocks for each output."""
        return {
            name: block_cls(
                input_shape=feature_dim,
                output_dim=num_actions if output_dim is None else output_dim,
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config, output_dim) in config.heads.items()
        }

    def __call__(self, inputs: dict[str, Float[Array, "B ..."]]) -> dict[str, Array]:
        """Forward pass through the network.

        Args:
            inputs: Dict mapping input names to input arrays

        Returns:
            Dict mapping head names to output arrays
        """
        # Encode each input
        encoded = []
        for name in sorted(self.encoders.keys()):
            x = self.encoders[name](inputs[name])  # [B, encoder_dim]
            encoded.append(x)

        # Concatenate and pass through trunk
        features = jnp.concatenate(encoded, axis=-1)  # [B, encoder_dim * num_inputs]
        features = self.trunk(features)  # [B, trunk_dim]

        # Generate outputs for each head
        outputs = {name: head(features) for name, head in self.heads.items()}

        return outputs

    def train_forward(self, data: RolloutData):
        """Forward pass for training with sequential data."""

        # Get shape from obs
        obs = data.trajectory.obs
        batch_size, length = obs.shape[0], obs.shape[1]

        inputs = {}
        for name in self.encoders.keys():
            x = getattr(data.trajectory, name)
            inputs[name] = x.reshape(-1, *x.shape[2:])  # Flatten [B, T, ...] -> [B*T, ...]

        out = self(inputs)

        # Reshape to original
        out = jax.tree.map(lambda x: x.reshape(batch_size, length, *x.shape[1:]), out)
        return out
