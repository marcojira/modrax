import jax.numpy as jnp
import pytest
from flax import nnx

from modrax.alg.base import OptimizerConfig, create_optimizer


class _Network(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.ones((2, 2)))


@pytest.mark.parametrize("optimizer_type", ["adam", "adamw", "radam", "sgd", "rmsprop", "muon"])
def test_create_optimizer(optimizer_type):
    optimizer = create_optimizer(_Network(), OptimizerConfig(optimizer_type=optimizer_type))

    assert isinstance(optimizer, nnx.Optimizer)


def test_lr_decay_requires_total_updates():
    with pytest.raises(AssertionError, match="total_num_updates"):
        create_optimizer(_Network(), OptimizerConfig(lr_decay=True))


def test_multiple_optimizer_state_roundtrip():
    networks = (_Network(), _Network(), _Network())
    optimizers = tuple(create_optimizer(network, OptimizerConfig()) for network in networks)

    graphdef, state = nnx.split((networks, *optimizers))
    restored = nnx.merge(graphdef, state)

    assert len(restored) == 4
