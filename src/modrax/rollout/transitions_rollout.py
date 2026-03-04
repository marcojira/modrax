from typing import Any, Callable

import jax
from flax import nnx, struct
from jaxtyping import Array, Float, Key

from modrax.env import StateWithMetrics
from modrax.network.base import Network


@struct.dataclass
class Transition:
    """A single (s, a, s') transition collected per step. Stacked over T steps by transitions_rollout."""

    obs: Float[Array, "B ..."]
    next_obs: Float[Array, "B ..."]
    info: Any
    actions: Float[Array, "B ..."]
    rewards: Float[Array, " B"]
    action_masks: Float[Array, "B A"]
    network_output: Any
    dones: Float[Array, " B"]
    truncations: Float[Array, " B"]
    episode_returns: Float[Array, " B"]
    episode_lengths: Float[Array, " B"]


def transitions_rollout(
    network: Network,
    step_fn: Callable,
    env_state: StateWithMetrics,
    num_steps: int,
    key: Key[Array, ""],
) -> tuple[StateWithMetrics, Transition]:
    """Collect num_steps transitions from B parallel envs.

    Unlike trajectory_rollout, this stores both obs and next_obs per step and
    does NOT reset network state on episode boundaries (suited for off-policy methods).
    Returned Transition arrays have shape [T, B, ...].

    Args:
        network: Policy network.
        step_fn: env.step — called as step_fn(state, action, keys).
        env_state: Initial batched env state with shape B.
        num_steps: Number of environment steps to collect.
        key: Single PRNG key.

    Returns:
        final_env_state: Env state after the last step.
        trajectory: Transition of shape [T, B, ...]
    """

    def step(carry, step_key):
        network, env_state = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, env_key = jax.random.split(step_key)

        # Run network
        action, out = network.policy(env_state, policy_key)

        # Step environment
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = step_fn(env_state, action, env_keys)

        trajectory = Transition(
            obs=obs,
            next_obs=new_env_state.obs,
            info=env_state.info,
            actions=action,
            rewards=step_output.reward,
            action_masks=action_mask,
            network_output=out,
            dones=step_output.done,
            truncations=step_output.truncation,
            episode_returns=env_state.episode_return + step_output.reward,
            episode_lengths=env_state.episode_length + 1,
        )

        return (network, new_env_state), trajectory

    step_keys = jax.random.split(key, num_steps)
    (_, final_env_state), trajectory = nnx.scan(step)((network, env_state), step_keys)

    return (
        final_env_state,
        trajectory,
    )


jit_transitions_rollout = nnx.jit(transitions_rollout, static_argnames=["step_fn", "num_steps"])
