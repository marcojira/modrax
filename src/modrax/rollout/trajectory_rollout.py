from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float, Key

from modrax.env import StateWithMetrics
from modrax.network.base import Network


@struct.dataclass
class Trajectory:
    """A sequence of experience collected over T steps from B parallel envs. Returned in (B, T, ...) order."""

    obs: Float[Array, "T B ..."]
    info: Any
    actions: Float[Array, "T B ..."]
    rewards: Float[Array, "T B"]
    action_masks: Float[Array, "T B A"]
    network_output: Any
    dones: Float[Array, "T B"]
    episode_returns: Float[Array, "T B"]
    episode_lengths: Float[Array, "T B"]


def trajectory_rollout(
    network: Network,
    step_fn: Callable,
    env_state: StateWithMetrics,
    num_steps: int,
    key: Key[Array, ""],
    random_action=False,
) -> tuple[StateWithMetrics, Trajectory]:
    """Collect num_steps of experience from B parallel envs for on-policy training.

    Resets network hidden state on episode boundaries. Output arrays are transposed
    to (B, T, ...) order for convenient per-env processing.

    Args:
        network: Policy network (state is reset when an env terminates).
        step_fn: env.step — called as step_fn(state, action, keys).
        env_state: Initial batched env state with shape B.
        num_steps: Number of environment steps to collect.
        key: Single PRNG key.
        random_action: If True, sample uniformly from valid actions (ignoring the network policy).

    Returns:
        final_env_state: Env state after the last step.
        trajectory: Trajectory with arrays in (B, T, ...) order.
    """

    def step(carry, step_key):
        network, env_state = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, env_key = jax.random.split(step_key)

        # Run network
        action, out = network.policy(env_state, policy_key)
        if random_action:
            uniform_logits = jnp.where(action_mask, 0.0, -jnp.inf)
            action = jax.random.categorical(key, uniform_logits)

        # Step environment
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = step_fn(env_state, action, env_keys)
        network.reset(step_output.done)  # Reset network state based on environments that terminated

        trajectory = Trajectory(
            obs=obs,
            info=env_state.info,
            actions=action,
            rewards=step_output.reward,
            action_masks=action_mask,
            network_output=out,
            dones=step_output.done,
            episode_returns=env_state.episode_return + step_output.reward,
            episode_lengths=env_state.episode_length + 1,
        )

        return (network, new_env_state), trajectory

    step_keys = jax.random.split(key, num_steps)
    (_, final_env_state), trajectory = nnx.scan(step)((network, env_state), step_keys)

    trajectory = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectory
    )  # Transpose to (B, T, ...)

    return final_env_state, trajectory
