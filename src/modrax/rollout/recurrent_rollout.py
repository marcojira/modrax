from typing import Callable, Protocol

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Key

from modrax.env import StateWithMetrics
from modrax.network.block.base import RecurrentState
from modrax.network.block.gtrxl import GTrXLRecurrentState
from modrax.rollout.base import NetworkInput, NetworkOutput, RolloutConfig, RolloutData, Trajectory


@struct.dataclass
class RecurrentRolloutData(RolloutData):
    recurrent_state: RecurrentState
    init_recurrent_state: RecurrentState


class RecurrentNetwork(Protocol):
    def __call__(
        self, inputs: NetworkInput, recurrent_state: RecurrentState
    ) -> tuple[NetworkOutput, RecurrentState]: ...


def recurrent_rollout(
    network: RecurrentNetwork,
    policy_fn: Callable,
    step_fn: Callable,
    env_state: StateWithMetrics,
    recurrent_state: RecurrentState,
    config: RolloutConfig,
    key: Key[Array, ""],
) -> tuple[StateWithMetrics, RecurrentState, RecurrentRolloutData]:
    def step(carry, step_key):
        network, env_state, prev_recurrent_state = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, env_key = jax.random.split(step_key)

        # Run network
        out, recurrent_state = network({"obs": obs}, prev_recurrent_state)
        action = policy_fn(out["policy"], action_mask, policy_key)

        # Step
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = step_fn(env_state, action, env_keys)

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

        recurrent_state = network.reset_recurrent_state(recurrent_state, step_output.done)

        # For GTRXL, don't store full recurrent state to save on memory
        if isinstance(recurrent_state, GTrXLRecurrentState):
            output_recurrent_state = GTrXLRecurrentState(
                memory=recurrent_state.memory[:, -1], mask=prev_recurrent_state.mask
            )
        else:
            output_recurrent_state = prev_recurrent_state

        return (network, new_env_state, recurrent_state), (trajectory, output_recurrent_state)

    step_keys = jax.random.split(key, config.num_steps)
    (_, final_env_state, final_recurrent_state), (trajectory, output_recurrent_state) = nnx.scan(
        step,
    )((network, env_state, recurrent_state), step_keys)

    final_out, _ = network({"obs": final_env_state.obs}, final_recurrent_state)

    data = RecurrentRolloutData(
        trajectory=jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), trajectory
        ),  # Transpose to (B, T, ...),
        final_out=final_out,
        recurrent_state=jax.tree.map(
            lambda x: jnp.swapaxes(x, 0, 1), output_recurrent_state
        ),  # Transpose to (B, T, ...),
        init_recurrent_state=recurrent_state,
    )

    return final_env_state, final_recurrent_state, data
