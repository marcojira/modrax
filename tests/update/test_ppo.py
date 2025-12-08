import jax
import jax.numpy as jnp
from flax import nnx

from modrax.policy import softmax_policy
from modrax.rollout.rollout import rollout
from modrax.update.ppo import PPOConfig, ppo_update


def test_ppo_update(env, network, optimizer):
    """Test PPO update using rollout-generated data."""
    key = jax.random.PRNGKey(0)
    reset_key, rollout_key, update_key = jax.random.split(key, 3)

    num_envs = 4
    num_steps = 8
    minibatch_size = 4

    reset_keys = jax.random.split(reset_key, num_envs)
    env_state = env.reset(reset_keys)

    final_state, data = rollout(
        network=network,
        policy_fn=softmax_policy,
        step_fn=env.step,
        env_state=env_state,
        num_steps=num_steps,
        key=rollout_key,
    )

    params_before = jax.tree.map(lambda x: x.copy(), nnx.state(network, nnx.Param))

    ppo_config = PPOConfig(minibatch_size=minibatch_size)
    loss, infos = ppo_update(
        network=network,
        optimizer=optimizer,
        data=data,
        final_state=final_state,
        config=ppo_config,
        key=update_key,
    )

    assert loss.shape == (num_envs // minibatch_size,)

    assert not jnp.any(jnp.isnan(loss)), "Loss contains NaN values"
    for key, value in infos.items():
        assert not jnp.any(jnp.isnan(value)), f"{key} contains NaN values"

    # Check if params were updated
    params_after = nnx.state(network, nnx.Param)
    params_changed = jax.tree.map(
        lambda before, after: not jnp.allclose(before, after),
        params_before,
        params_after,
    )
    assert any(jax.tree.leaves(params_changed)), "Network parameters were not updated"
