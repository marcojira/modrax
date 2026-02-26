from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int, Key

from modrax.env import StateWithMetrics
from modrax.network.base import Network


def eval_rollout(
    network: Network,
    step_fn: Callable,
    env_state: StateWithMetrics,
    key: Key[Array, ""],
) -> tuple[Float[Array, " B"], Int[Array, " B"]]:
    """Run until all envs have terminated at least once, return episode returns and lengths."""
    num_envs = env_state.obs.shape[0]
    init_val = (
        network,
        env_state,
        jnp.zeros(num_envs, dtype=jnp.bool_),
        jnp.zeros(num_envs),
        jnp.zeros(num_envs, dtype=jnp.int32),
        key,
    )

    def cond_fn(val):
        _, _, done_mask, _, _, _ = val
        return ~done_mask.all()

    def body_fn(val):
        network, env_state, done_mask, ep_returns, ep_lengths, key = val
        key, policy_key, env_key = jax.random.split(key, 3)

        action, _ = network.policy(env_state, policy_key)

        env_keys = jax.random.split(env_key, num_envs)
        step_output, new_env_state = step_fn(env_state, action, env_keys)

        # Capture returns/lengths at first termination
        first_done = step_output.done & ~done_mask
        current_return = env_state.episode_return + step_output.reward
        current_length = env_state.episode_length + 1

        ep_returns = jnp.where(first_done, current_return, ep_returns)
        ep_lengths = jnp.where(first_done, current_length, ep_lengths)
        done_mask = done_mask | step_output.done

        return (network, new_env_state, done_mask, ep_returns, ep_lengths, key)

    _, _, _, episode_returns, episode_lengths, _ = nnx.while_loop(cond_fn, body_fn, init_val)
    return episode_returns, episode_lengths
