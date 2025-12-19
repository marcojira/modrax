"""AlphaZero network architecture."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float

from modrax.network.base import Network, NetworkConfig
from modrax.rollout.base import RolloutData
from modrax.types import Shape


class AZNetConfig(NetworkConfig):
    num_channels: int = 32
    num_blocks: int = 3
    resnet_v2: bool = True


class ResBlockV1(nnx.Module):
    """ResNet v1 block: Conv → BN → ReLU → Conv → BN → (+ residual) → ReLU."""

    def __init__(self, num_channels: int, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(
            num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)
        self.conv2 = nnx.Conv(
            num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B H W C"]:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return nnx.relu(x + residual)


class ResBlockV2(nnx.Module):
    """ResNet v2 block: BN → ReLU → Conv → BN → ReLU → Conv → (+ residual)."""

    def __init__(self, num_channels: int, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)
        self.conv1 = nnx.Conv(
            num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)
        self.conv2 = nnx.Conv(
            num_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )

    def __call__(self, x: Float[Array, "B H W C"]) -> Float[Array, "B H W C"]:
        residual = x
        x = self.bn1(x)
        x = nnx.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = nnx.relu(x)
        x = self.conv2(x)
        return x + residual


class AZNet(Network):
    """AlphaZero network architecture."""

    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        config: AZNetConfig,
        rngs: nnx.Rngs,
    ):
        h, w, in_channels = obs_shape
        num_channels = config.num_channels
        self.resnet_v2 = config.resnet_v2

        # Initial conv
        self.initial_conv = nnx.Conv(
            in_channels, num_channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )

        # Optional initial BN for v1
        if not config.resnet_v2:
            self.initial_bn = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)

        # ResNet blocks
        block_cls = ResBlockV2 if config.resnet_v2 else ResBlockV1
        self.blocks = [block_cls(num_channels, rngs) for _ in range(config.num_blocks)]

        # Final BN for v2
        if config.resnet_v2:
            self.final_bn = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)

        # Policy head
        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=(1, 1), rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, rngs=rngs)
        self.policy_linear = nnx.Linear(h * w * 2, num_actions, rngs=rngs)

        # Value head
        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=(1, 1), rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, rngs=rngs)
        self.value_linear1 = nnx.Linear(h * w, num_channels, rngs=rngs)
        self.value_linear2 = nnx.Linear(num_channels, 1, rngs=rngs)

    def __call__(self, inputs: dict[str, Float[Array, "B ..."]]) -> dict[str, Array]:
        """Forward pass returning policy logits and value."""
        x = inputs["obs"]
        x = x.astype(jnp.float32)
        x = self.initial_conv(x)

        if not self.resnet_v2:
            x = self.initial_bn(x)
            x = nnx.relu(x)

        for block in self.blocks:
            x = block(x)

        if self.resnet_v2:
            x = self.final_bn(x)
            x = nnx.relu(x)

        # Policy head
        logits = self.policy_conv(x)
        logits = self.policy_bn(logits)
        logits = nnx.relu(logits)
        logits = logits.reshape(logits.shape[0], -1)
        logits = self.policy_linear(logits)

        # Value head
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = nnx.relu(v)
        v = v.reshape(v.shape[0], -1)
        v = self.value_linear1(v)
        v = nnx.relu(v)
        v = self.value_linear2(v)
        v = jnp.tanh(v)

        return {"value": v, "policy": logits}

    def train_forward(self, data: RolloutData) -> dict[str, Array]:
        """Forward pass for training with sequential data."""
        obs = data.trajectory.obs
        batch_size, length = obs.shape[0], obs.shape[1]

        inputs = {"obs": obs.reshape(-1, *obs.shape[2:])}
        out = self(inputs)

        out = jax.tree.map(lambda x: x.reshape(batch_size, length, *x.shape[1:]), out)
        return out
