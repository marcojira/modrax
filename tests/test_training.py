from types import SimpleNamespace

import jax
import jax.numpy as jnp
from flax import nnx

from modrax.logging import WandbConfig
from modrax.network.base import Network
from modrax.training import train


class SavingNetwork(Network):
    saved = []

    def __init__(self, value: float):
        """Initialize a network with one scalar parameter."""
        self.value = nnx.Param(jnp.array(value))

    def save(self, path: str):
        """Record the object and path selected for checkpointing."""
        self.saved.append((self, path))


def test_train_saves_and_returns_current_network(tmp_path):
    """The final checkpoint must use the network held by algorithm state."""
    original_network = SavingNetwork(0.0)
    trained_network = SavingNetwork(1.0)
    algorithm = SimpleNamespace(
        network=original_network,
        get_network=lambda: trained_network,
        cfg=SimpleNamespace(total_steps=0),
        env=None,
        env_steps_per_epoch=1,
    )
    config = SimpleNamespace(
        display_network=False,
        wandb=WandbConfig(),
        save_path=str(tmp_path),
    )

    SavingNetwork.saved.clear()
    result = train(algorithm, config, jax.random.key(0))

    assert result is trained_network
    assert SavingNetwork.saved == [(trained_network, str(tmp_path / "checkpoint"))]
