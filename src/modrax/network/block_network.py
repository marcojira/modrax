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
        str, tuple[type[Block], BlockConfig]
    ]  # Mapping from input name to (BlockClass, BlockConfig)
    encoder_dim: int  # Output dimension for all encoders

    trunk: tuple[type[Block], BlockConfig]
    trunk_dim: int  # Output dimension for trunk

    heads: dict[
        str, tuple[type[Block], BlockConfig]
    ]  # Mapping output name to (BlockClass, BlockConfig)


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
        >>> from modrax.network.component.linear import Linear, LinearConfig
        >>> from modrax.network.component.mlp import MLP, MLPConfig
        >>> import jax.nn as jnn
        >>>
        >>> config = BlockNetworkConfig(
        ...     encoders={"obs": (Linear, LinearConfig())},
        ...     encoder_dim=32,
        ...     trunk=(MLP, MLPConfig(hidden_dims=(64, 64), activation_fn=jnn.relu)),
        ...     trunk_dim=64,
        ...     heads={
        ...         "policy": (MLP, MLPConfig(hidden_dims=(32,), activation_fn=jnn.relu)),
        ...         "value": (Linear, LinearConfig()),
        ...     },
        ... )
        >>> network = BlockNetwork(
        ...     input_shapes={"obs": (4,)},
        ...     output_dims={"policy": 4, "value": 1},
        ...     config=config,
        ...     rngs=rngs,
        ... )
        >>> outputs = network({"obs": obs_batch})
        >>> # outputs = {"policy": [B, 4], "value": [B, 1]}
    """

    def __init__(
        self,
        input_shapes: dict[str, Shape | int],  # Dict mapping input names to shapes
        output_dims: dict[str, int],  # Dict mapping head names to output dimensions
        config: BlockNetworkConfig,
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

        # Build trunk block (takes concatenated encoders)
        trunk_cls, trunk_config = config.trunk
        trunk_input_dim = len(input_shapes) * config.encoder_dim
        self.trunk = trunk_cls(
            input_shape=trunk_input_dim,
            output_dim=config.trunk_dim,
            config=trunk_config,
            rngs=rngs,
        )

        # Build head block for each output
        self.heads = {
            name: block_cls(
                input_shape=config.trunk_dim,
                output_dim=output_dims[name],
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config) in config.heads.items()
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
