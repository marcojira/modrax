from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, Key

from modrax.env.base import StateWithMetrics
from modrax.network.base import Network


def eval_rollout(
    network: Network,
    step_fn: Callable,
    env_state: StateWithMetrics,
    key: Key[Array, ""],
    max_steps: int = 1000,
) -> tuple[Float[Array, " B"], Int[Array, " B"], StateWithMetrics]:
    """Scan for max_steps, return episode returns, lengths, and full trajectories."""
    num_envs = env_state.obs.shape[0]

    def step(carry, step_key):
        network, env_state, done_mask, ep_returns, ep_lengths = carry
        policy_key, env_key = jax.random.split(step_key)

        action, _ = network.policy(env_state, policy_key)
        env_keys = jax.random.split(env_key, num_envs)
        step_output, new_env_state = step_fn(env_state, action, env_keys)

        first_done = step_output.done & ~done_mask
        current_return = env_state.episode_return + step_output.reward
        current_length = env_state.episode_length + 1

        ep_returns = jnp.where(first_done, current_return, ep_returns)
        ep_lengths = jnp.where(first_done, current_length, ep_lengths)
        done_mask = done_mask | step_output.done

        return (network, new_env_state, done_mask, ep_returns, ep_lengths), new_env_state

    step_keys = jax.random.split(key, max_steps)
    (_, _, _, episode_returns, episode_lengths), trajectories = nnx.scan(step)(
        (
            network,
            env_state,
            jnp.zeros(num_envs, dtype=jnp.bool_),
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
        ),
        step_keys,
    )

    return episode_returns, episode_lengths, trajectories
