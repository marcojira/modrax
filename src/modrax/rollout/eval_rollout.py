import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.env.base import Env, StateWithMetrics
from modrax.network.base import Network


@nnx.jit(static_argnames=["env", "num_envs", "max_steps", "num_trajectories"])
def eval_rollout(
    env: Env,
    network: Network,
    num_envs: int,
    key: Key[Array, ""],
    max_steps: int = 1000,
    num_trajectories: int = 10,
) -> tuple[dict[str, Float[Array, ""]], StateWithMetrics]:
    """Run evaluation episodes and return mean metrics + trajectories.

    Resets the env, runs up to max_steps, and tracks the first completed episode per env.
    If an env never finishes, its accumulated return/length at max_steps is used.
    """
    eval_env_state = env.reset(jax.random.split(key, num_envs))

    def step(carry, step_key):
        network, env_state, done_mask, ep_returns, ep_lengths = carry
        policy_key, env_key = jax.random.split(step_key)

        action, _ = network.policy(env_state, policy_key)
        env_keys = jax.random.split(env_key, num_envs)
        step_output, new_env_state = env.step(env_state, action, env_keys)
        network.reset(step_output.done)

        first_done = step_output.done & ~done_mask
        current_return = env_state.episode_return + step_output.reward
        current_length = env_state.episode_length + 1

        ep_returns = jnp.where(first_done, current_return, ep_returns)
        ep_lengths = jnp.where(first_done, current_length, ep_lengths)
        done_mask = done_mask | step_output.done

        return (network, new_env_state, done_mask, ep_returns, ep_lengths), jax.tree.map(
            lambda x: x[:num_trajectories], new_env_state
        )

    step_keys = jax.random.split(key, max_steps)
    (_, final_env_state, done_mask, episode_returns, episode_lengths), trajectories = nnx.scan(
        step
    )(
        (
            network,
            eval_env_state,
            jnp.zeros(num_envs, dtype=jnp.bool_),
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
        ),
        step_keys,
    )

    # For envs that never finished, use their accumulated return/length
    episode_returns = jnp.where(done_mask, episode_returns, final_env_state.episode_return)
    episode_lengths = jnp.where(done_mask, episode_lengths, final_env_state.episode_length)

    metrics = {
        "eval_return": episode_returns.mean(),
        "eval_length": episode_lengths.mean(),
    }
    return metrics, trajectories
