import jax
import jax.numpy as jnp

from modrax.network.block.linear import Linear, LinearConfig
from modrax.network.block.mlp import MLP, MLPConfig
from modrax.network.block_network import BlockNetwork, BlockNetworkConfig


def test_block_network(env, rngs, batch_obs):
    """Test BlockNetwork with multiple inputs and outputs."""
    config = BlockNetworkConfig(
        encoders={
            "obs": (MLP, MLPConfig(hidden_dims=(32,), activation_fn=jax.nn.relu), None),
            "prev_action": (Linear, LinearConfig(), (env.num_actions,)),
        },
        encoder_dim=32,
        trunk=(MLP, MLPConfig(hidden_dims=(64, 64), activation_fn=jax.nn.relu)),
        trunk_dim=64,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(32,), activation_fn=jax.nn.relu), None),
            "value": (Linear, LinearConfig(), 1),
        },
    )

    network = BlockNetwork(
        obs_shape=env.obs_shape,
        num_actions=env.num_actions,
        config=config,
        rngs=rngs,
    )

    # Create dummy prev_action (one-hot)
    prev_action = jnp.zeros((batch_obs.shape[0], env.num_actions))

    output = network({"obs": batch_obs, "prev_action": prev_action})

    assert isinstance(output, dict)
    assert set(output.keys()) == {"policy", "value"}
    assert output["policy"].shape == (batch_obs.shape[0], env.num_actions)
    assert output["value"].shape == (batch_obs.shape[0], 1)
