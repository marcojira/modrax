import jax
from flax import nnx

from modrax.network.az_net import AZNet, AZNetConfig


def test_az_net_forward():
    """Test AZNet forward pass."""
    obs_shape = (3, 3, 2)
    num_actions = 9
    batch_obs = jax.random.normal(jax.random.key(0), (4, *obs_shape))

    config = AZNetConfig(num_channels=16, num_blocks=2)
    network = AZNet(obs_shape, num_actions, config, nnx.Rngs(0))

    out = network({"obs": batch_obs})

    assert out["policy"].shape == (4, num_actions)
    assert out["value"].shape == (4, 1)
