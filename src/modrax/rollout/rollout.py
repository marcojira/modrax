from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Key

from modrax.env import StateWithMetrics
from modrax.policy import PolicyFn
from modrax.rollout.base import NetworkInput, NetworkOutput, RolloutConfig, RolloutData, Trajectory


class ForwardNetwork(Protocol):
    def __call__(self, inputs: NetworkInput) -> NetworkOutput: ...


def rollout(
    network: ForwardNetwork,
    policy_fn: PolicyFn,
    step_fn: Callable,
    env_state: StateWithMetrics,
    recurrent_state: None,
    config: RolloutConfig,
    key: Key[Array, ""],
) -> tuple[StateWithMetrics, None, RolloutData]:
    def step(carry, step_key):
        network, env_state = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, env_key = jax.random.split(step_key)

        # Run network
        out = network({"obs": obs})
        logits = out["policy"]
        action = policy_fn(logits, action_mask, policy_key)

        # Step environment
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = step_fn(env_state, action, env_keys)

        trajectory = Trajectory(
            obs=obs,
            actions=action,
            rewards=step_output.reward,
            action_masks=action_mask,
            network_output=out,
            dones=step_output.done,
            episode_returns=env_state.episode_return + step_output.reward,
            episode_lengths=env_state.episode_length + 1,
        )

        return (network, new_env_state), trajectory

    step_keys = jax.random.split(key, config.num_steps)
    (_, final_env_state), trajectory = nnx.scan(step)((network, env_state), step_keys)

    final_out = network({"obs": final_env_state.obs})

    data = RolloutData(
        trajectory=jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), trajectory
        ),  # Transpose to (B, T, ...)
        final_out=final_out,
    )

    return final_env_state, None, data
