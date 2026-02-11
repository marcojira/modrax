"""Simple CNN block matching MRQ architecture."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from jaxtyping import Array, Float
from pydantic import ConfigDict

from modrax.network.block.base import Block, BlockConfig
from modrax.types import Shape

# Xavier uniform with ReLU gain (sqrt(2)) to match PyTorch reference
KERNEL_INIT = initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
BIAS_INIT = initializers.zeros


class CNNConfig(BlockConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    channels: tuple[int, ...] = (32, 32, 32, 32)
    kernel_sizes: tuple[int, ...] = (3, 3, 3, 3)
    strides: tuple[int, ...] = (2, 2, 2, 1)
    activation_fn: Callable = jax.nn.elu
    normalize_input: bool = False
    layer_norm: bool = True


def compute_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: str) -> int:
    """Compute output size for a single conv layer."""
    if padding == "SAME":
        return (input_size + stride - 1) // stride
    elif padding == "VALID":
        return (input_size - kernel_size) // stride + 1
    else:
        raise NotImplementedError(f"Padding {padding} not supported")


class CNN(Block):
    """Simple CNN encoder matching MRQ architecture.

    Architecture:
    - Input normalization (x/255 - 0.5)
    - 4 Conv2d layers with configurable channels, kernels, strides
    - Activation after each conv
    - Flatten
    - Linear projection with optional layer norm + activation
    """

    def __init__(
        self,
        input_shape: Shape | int,
        output_dim: int,
        config: CNNConfig,
        rngs: nnx.Rngs,
    ):
        if isinstance(input_shape, int):
            raise ValueError("CNN requires full input_shape (H, W, C), not just channels")

        h, w, in_channels = input_shape
        self.config = config

        # Validate config
        if not (len(config.channels) == len(config.kernel_sizes) == len(config.strides)):
            raise ValueError("channels, kernel_sizes, and strides must have same length")

        # Create conv layers
        self.convs = []
        current_channels = in_channels
        current_h, current_w = h, w

        for out_channels, kernel_size, stride in zip(
            config.channels, config.kernel_sizes, config.strides
        ):
            self.convs.append(
                nnx.Conv(
                    current_channels,
                    out_channels,
                    kernel_size=(kernel_size, kernel_size),
                    strides=(stride, stride),
                    padding="SAME",
                    kernel_init=KERNEL_INIT,
                    bias_init=BIAS_INIT,
                    rngs=rngs,
                )
            )
            current_channels = out_channels
            current_h = compute_conv_output_size(current_h, kernel_size, stride, "SAME")
            current_w = compute_conv_output_size(current_w, kernel_size, stride, "SAME")

        # Compute flattened dimension
        flattened_dim = current_h * current_w * current_channels

        # Linear projection
        self.proj = nnx.Linear(
            flattened_dim, output_dim, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT, rngs=rngs
        )

        # Optional layer norm for final projection (no learnable affine to match reference)
        if config.layer_norm:
            self.layer_norm = nnx.LayerNorm(output_dim, use_scale=False, use_bias=False, rngs=rngs)

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B D"]:
        # Input normalization
        if self.config.normalize_input:
            x = x / 255.0 - 0.5

        # Apply conv layers with activation
        for conv in self.convs:
            x = conv(x)
            x = self.config.activation_fn(x)

        # Flatten
        x = x.reshape(x.shape[0], -1)

        # Linear projection
        x = self.proj(x)

        # Layer norm + activation on final output
        if self.config.layer_norm:
            x = self.layer_norm(x)
        x = self.config.activation_fn(x)

        return x
