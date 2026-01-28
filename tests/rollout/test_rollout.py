import jax

from modrax.policy import softmax_policy
from modrax.rollout.rollout import rollout


def test_rollout(env, network):
    key = jax.random.PRNGKey(0)
    reset_key, rollout_key = jax.random.split(key)

    num_envs = 4
    reset_keys = jax.random.split(reset_key, num_envs)
    env_state = env.reset(reset_keys)

    num_steps = 10
    final_state, recurrent_state, data = rollout(
        network=network,
        policy_fn=softmax_policy,
        step_fn=env.step,
        env_state=env_state,
        recurrent_state=None,
        num_steps=num_steps,
        key=rollout_key,
    )
    assert recurrent_state is None

    traj = data.trajectory
    assert traj.obs.shape[0] == num_envs
    assert traj.obs.shape[1] == num_steps
    assert traj.actions.shape == (num_envs, num_steps)
    assert traj.rewards.shape == (num_envs, num_steps)
    assert traj.dones.shape == (num_envs, num_steps)
