from typing import Any, Callable

import jax
from flax import nnx, struct
from jaxtyping import Array, Float, Key

from modrax.env import StateWithMetrics
from modrax.network.base import Network


@struct.dataclass
class Transition:
    obs: Float[Array, "B ..."]
    next_obs: Float[Array, "B ..."]
    info: Any
    actions: Float[Array, "B ..."]
    rewards: Float[Array, " B"]
    action_masks: Float[Array, "B A"]
    network_output: Any
    dones: Float[Array, " B"]
    truncations: Float[Array, " B"]
    episode_returns: Float[Array, "T B"]
    episode_lengths: Float[Array, "T B"]


def transitions_rollout(
    network: Network,
    step_fn: Callable,
    env_state: StateWithMetrics,
    num_steps: int,
    key: Key[Array, ""],
) -> tuple[StateWithMetrics, Transition]:
    def step(carry, step_key):
        network, env_state = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, env_key = jax.random.split(step_key)

        # Run network
        action, out = network.get_action(obs, policy_key)

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
