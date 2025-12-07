import jax

from modrax.network.block.mlp import MLP, MLPConfig
from modrax.network.block_network import BlockNetwork, BlockNetworkConfig
from modrax.policy import softmax_policy
from modrax.rollout.rollout import rollout


def test_rollout(env, rngs):
    key = jax.random.PRNGKey(0)
    reset_key, rollout_key = jax.random.split(key)

    num_envs = 4
    reset_keys = jax.random.split(reset_key, num_envs)
    env_state = env.reset(reset_keys)

    config = BlockNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(32,)))},
        encoder_dim=32,
        trunk=(MLP, MLPConfig(hidden_dims=(64,))),
        trunk_dim=64,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(32,))),
            "value": (MLP, MLPConfig(hidden_dims=(32,))),
        },
    )
    network = BlockNetwork(
        input_shapes={"obs": env.obs_shape},
        output_dims={"policy": env.num_actions, "value": 1},
        config=config,
        rngs=rngs,
    )

    num_steps = 10
    final_state, data = rollout(
        network=network,
        policy_fn=softmax_policy,
        step_fn=env.step,
        env_state=env_state,
        num_steps=num_steps,
        key=rollout_key,
    )

    traj = data.trajectory
    assert traj.observations.shape[0] == num_steps
    assert traj.observations.shape[1] == num_envs
    assert traj.actions.shape == (num_steps, num_envs)
    assert traj.rewards.shape == (num_steps, num_envs)
    assert traj.dones.shape == (num_steps, num_envs)