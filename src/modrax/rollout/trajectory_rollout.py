from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Bool, Float, Key

from modrax.env import StateWithMetrics
from modrax.network.base import Network


@struct.dataclass
class Trajectory:
    """A sequence of experience collected over T steps from B parallel envs.
    Returned in (B, T, ...) order.
    """

    obs: Float[Array, "B T ..."]
    info: Any
    actions: Float[Array, "B T ..."]
    rewards: Float[Array, "B T"]
    action_masks: Float[Array, "B T A"]
    network_output: Any
    dones: Float[Array, "B T"]
    # Marks steps before per-environment termination in episodic mode.
    valid_mask: Bool[Array, "B T"]
    episode_returns: Float[Array, "B T"]
    episode_lengths: Float[Array, "B T"]


def trajectory_rollout(
    network: Network,
    step_fn: Callable,
    env_state: StateWithMetrics,
    num_steps: int,
    key: Key[Array, ""],
    random_action=False,
    episodic=False,
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
        episodic: If True, `valid_mask` is False for any step taken after a per-env
            terminal transition. Intended for use with envs configured with
            auto_reset=False so episodes actually end. If False, `valid_mask` is
            all True.

    Returns:
        final_env_state: Env state after the last step.
        trajectory: Trajectory with arrays in (B, T, ...) order.
    """

    def step(carry, step_key):
        network, env_state, alive = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, random_action_key, env_key = jax.random.split(step_key, 3)

        # Run network
        action, out = network.policy(env_state, policy_key)
        if random_action:
            uniform_logits = jnp.where(action_mask, 0.0, -jnp.inf)
            action = jax.random.categorical(random_action_key, uniform_logits)

        # Step environment
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = step_fn(env_state, action, env_keys)
        network.reset(step_output.done)  # Reset network state based on environments that terminated

        valid_mask = alive if episodic else jnp.ones_like(alive)
        # In episodic mode envs stay in their terminal state, so step_output.done can
        # stay True for many steps. Only count the transition (first done) as terminal.
        dones = step_output.done & alive if episodic else step_output.done

        trajectory = Trajectory(
            obs=obs,
            info=env_state.info,
            actions=action,
            rewards=step_output.reward,
            action_masks=action_mask,
            network_output=out,
            dones=dones,
            valid_mask=valid_mask,
            episode_returns=env_state.episode_return + step_output.reward,
            episode_lengths=env_state.episode_length + 1,
        )

        new_alive = alive & ~step_output.done
        return (network, new_env_state, new_alive), trajectory

    step_keys = jax.random.split(key, num_steps)
    init_alive = jnp.ones(env_state.obs.shape[0], dtype=jnp.bool_)
    (_, final_env_state, final_alive), trajectory = nnx.scan(step)(
        (network, env_state, init_alive), step_keys
    )

    trajectory = jax.tree_util.tree_map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectory
    )  # Transpose to (B, T, ...)

    if episodic:
        # Truncate ongoing episodes by marking the last step as done for any env
        # still alive at the end of the rollout.
        new_dones = trajectory.dones.at[:, -1].set(trajectory.dones[:, -1] | final_alive)
        trajectory = trajectory.replace(dones=new_dones)

    return final_env_state, trajectory
