from typing import Callable, Protocol

import jax
from flax import nnx
from jaxtyping import Array, Key

from modrax.env import StateWithMetrics
from modrax.rollout.base import NetworkInput, NetworkOutput, Transition


class ForwardNetwork(Protocol):
    def __call__(self, inputs: NetworkInput) -> NetworkOutput: ...


def rollout(
    network: ForwardNetwork,
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
        action, _ = network.get_action(obs, policy_key)

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


jit_rollout = nnx.jit(rollout, static_argnames=["step_fn", "num_steps"])
