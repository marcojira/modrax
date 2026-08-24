import jax.numpy as jnp
from flax import nnx

from modrax.alg.base import OptimizerConfig, create_optimizer, update_network_minibatches


class _Network(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.array(0.0))


def test_create_optimizer_updates_parameters():
    network = _Network()
    optimizer = create_optimizer(network, OptimizerConfig(optimizer_type="sgd", learning_rate=0.1))

    def loss_fn(model, target, _config):
        loss = jnp.square(model.weight - target)
        return loss, {}

    losses, _ = update_network_minibatches(
        network, optimizer, jnp.array([1.0, 1.0]), loss_fn, config=None
    )

    assert losses.shape == (2,)
    assert network.weight[...] > 0
