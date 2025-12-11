"""IMPALA CNN block for visual observations."""

import math

from flax import nnx
from jaxtyping import Array, Float

from modrax.network.block.base import Block, BlockConfig
from modrax.types import Shape


class ImpalaCNNConfig(BlockConfig):
    channels: tuple[int, ...] = (64, 64, 128)
    use_batch_norm: bool = True


def compute_output_spatial_size(input_size: int, num_pools: int) -> int:
    """Compute spatial size after repeated stride-2 SAME pooling."""
    size = input_size
    for _ in range(num_pools):
        size = math.ceil(size / 2)
    return size


class ResNetBlock(nnx.Module):
    """Single ResNet block: ReLU → BN → Conv → ReLU → BN → Conv + skip."""

    def __init__(self, channels: int, rngs: nnx.Rngs, use_batch_norm: bool = True):
        self.use_batch_norm = use_batch_norm
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        if use_batch_norm:
            self.bn1 = nnx.BatchNorm(channels, rngs=rngs)
            self.bn2 = nnx.BatchNorm(channels, rngs=rngs)

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B H W C"]:
        residual = x
        x = nnx.relu(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        x = self.conv1(x)
        x = nnx.relu(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        x = self.conv2(x)
        return x + residual


class ConvStack(nnx.Module):
    """Single stack: BN → Conv → MaxPool → 2 ResNet blocks."""

    def __init__(
        self, in_channels: int, out_channels: int, rngs: nnx.Rngs, use_batch_norm: bool = True
    ):
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.bn = nnx.BatchNorm(in_channels, rngs=rngs)
        self.conv = nnx.Conv(
            in_channels, out_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.res_block1 = ResNetBlock(out_channels, rngs, use_batch_norm)
        self.res_block2 = ResNetBlock(out_channels, rngs, use_batch_norm)

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B H W C"]:
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.conv(x)
        x = nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding="SAME")
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class ImpalaCNN(Block):
    """IMPALA CNN encoder for visual observations."""

    def __init__(
        self,
        input_shape: Shape | int,
        output_dim: int,
        config: ImpalaCNNConfig,
        rngs: nnx.Rngs,
    ):
        if isinstance(input_shape, int):
            raise ValueError("ImpalaCNN requires full input_shape (H, W, C), not just channels")
        h, w, in_channels = input_shape

        self.stacks = []
        for out_channels in config.channels:
            self.stacks.append(ConvStack(in_channels, out_channels, rngs, config.use_batch_norm))
            in_channels = out_channels

        # Compute flattened size after all pooling operations
        num_pools = len(config.channels)
        h_out = compute_output_spatial_size(h, num_pools)
        w_out = compute_output_spatial_size(w, num_pools)
        flattened_dim = h_out * w_out * config.channels[-1]

        self.proj = nnx.Linear(flattened_dim, output_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B D"]:
        for stack in self.stacks:
            x = stack(x)
        x = nnx.relu(x)
        x = x.reshape(x.shape[0], -1)
        return self.proj(x)
