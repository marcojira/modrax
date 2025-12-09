import jax

from modrax.policy import softmax_policy
from modrax.rollout.base import RolloutConfig
from modrax.rollout.recurrent_rollout import recurrent_rollout


def test_recurrent_rollout(env, recurrent_network):
    key = jax.random.PRNGKey(0)
    reset_key, rollout_key = jax.random.split(key)

    num_envs = 4
    reset_keys = jax.random.split(reset_key, num_envs)
    env_state = env.reset(reset_keys)

    recurrent_state = recurrent_network.init_recurrent_state(num_envs)

    num_steps = 8
    final_state, final_recurrent_state, data = recurrent_rollout(
        network=recurrent_network,
        policy_fn=softmax_policy,
        step_fn=env.step,
        env_state=env_state,
        recurrent_state=recurrent_state,
        config=RolloutConfig(num_steps=num_steps),
        key=rollout_key,
    )

    traj = data.trajectory
    assert traj.obs.shape[0] == num_envs
    assert traj.obs.shape[1] == num_steps
    assert traj.actions.shape == (num_envs, num_steps)
    assert traj.rewards.shape == (num_envs, num_steps)
    assert traj.dones.shape == (num_envs, num_steps)
