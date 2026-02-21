"""Simple CNN encoder matching MRQ architecture."""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx import initializers
from jaxtyping import Array, Float

from modrax.types import Shape

# Xavier uniform with ReLU gain (sqrt(2)) to match PyTorch reference
KERNEL_INIT = initializers.variance_scaling(scale=2.0, mode="fan_avg", distribution="uniform")
BIAS_INIT = initializers.zeros


def compute_conv_output_size(input_size: int, kernel_size: int, stride: int, padding: str) -> int:
    """Compute output size for a single conv layer."""
    if padding == "SAME":
        return (input_size + stride - 1) // stride
    elif padding == "VALID":
        return (input_size - kernel_size) // stride + 1
    else:
        raise NotImplementedError(f"Padding {padding} not supported")


class CNN(nnx.Module):
    """Simple CNN encoder matching MRQ architecture.

    Architecture: Conv layers -> Flatten -> Linear projection -> LayerNorm -> Activation
    """

    def __init__(
        self,
        input_shape: Shape,
        output_dim: int,
        rngs: nnx.Rngs,
        channels: Sequence[int] = (32, 32, 32, 32),
        kernel_sizes: Sequence[int] = (3, 3, 3, 3),
        strides: Sequence[int] = (2, 2, 2, 1),
        activation_fn: Callable = jax.nn.elu,
        normalize_input: bool = False,
        layer_norm: bool = True,
    ):
        if len(input_shape) != 3:
            raise ValueError("CNN requires input_shape (H, W, C)")
        if not (len(channels) == len(kernel_sizes) == len(strides)):
            raise ValueError("channels, kernel_sizes, and strides must have same length")

        self.activation_fn = activation_fn
        self.normalize_input = normalize_input

        h, w, in_channels = input_shape
        current_h, current_w = h, w

        # Conv layers
        self.convs = []
        current_channels = in_channels
        for out_channels, kernel_size, stride in zip(channels, kernel_sizes, strides):
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

        flattened_dim = current_h * current_w * current_channels

        # Linear projection
        self.proj = nnx.Linear(
            flattened_dim, output_dim, kernel_init=KERNEL_INIT, bias_init=BIAS_INIT, rngs=rngs
        )

        # Optional layer norm (no learnable affine to match reference)
        self.ln = (
            nnx.LayerNorm(output_dim, use_scale=False, use_bias=False, rngs=rngs)
            if layer_norm
            else None
        )

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B D"]:
        if self.normalize_input:
            x = x / 255.0 - 0.5

        for conv in self.convs:
            x = self.activation_fn(conv(x))

        x = x.reshape(x.shape[0], -1)
        x = self.proj(x)

        if self.ln is not None:
            x = self.ln(x)
        x = self.activation_fn(x)

        return x
