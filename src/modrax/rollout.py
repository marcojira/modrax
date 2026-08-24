from typing import Any, Callable

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Bool, Float, Key, Shaped

from modrax.env import StateWithMetrics
from modrax.network.base import Network


@struct.dataclass
class Trajectory:
    """A sequence of experience collected over T steps from B parallel envs."""

    obs: Float[Array, "B T ..."]
    info: Any
    actions: Shaped[Array, "B T ..."]
    rewards: Float[Array, "B T"]
    action_masks: Float[Array, "B T A"]
    network_output: Any
    dones: Bool[Array, "B T"]
    truncations: Bool[Array, "B T"]
    valid_mask: Bool[Array, "B T"]
    episode_returns: Float[Array, "B T"]
    episode_lengths: Float[Array, "B T"]


@struct.dataclass
class Transition:
    """A batch of independently sampleable transitions in (B, T, ...) order."""

    obs: Float[Array, "B T ..."]
    next_obs: Float[Array, "B T ..."]
    info: Any
    actions: Shaped[Array, "B T ..."]
    rewards: Float[Array, "B T"]
    action_masks: Float[Array, "B T A"]
    network_output: Any
    dones: Bool[Array, "B T"]
    truncations: Bool[Array, "B T"]
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
    """Collect num_steps of experience from B parallel envs."""

    def step(carry, step_key):
        network, env_state, alive = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        policy_key, random_action_key, env_key = jax.random.split(step_key, 3)

        action, out = network.policy(env_state, policy_key)
        if random_action:
            uniform_logits = jnp.where(action_mask, 0.0, -jnp.inf)
            action = jax.random.categorical(random_action_key, uniform_logits)

        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = step_fn(env_state, action, env_keys)
        network.reset_episodes(step_output.done)

        valid_mask = alive if episodic else jnp.ones_like(alive)
        dones = step_output.done & alive if episodic else step_output.done

        trajectory = Trajectory(
            obs=obs,
            info=env_state.info,
            actions=action,
            rewards=step_output.reward,
            action_masks=action_mask,
            network_output=out,
            dones=dones,
            truncations=step_output.truncation,
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
    trajectory = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory)

    if episodic:
        new_dones = trajectory.dones.at[:, -1].set(trajectory.dones[:, -1] | final_alive)
        trajectory = trajectory.replace(dones=new_dones)

    return final_env_state, trajectory


def trajectory_to_transitions(trajectory: Trajectory, final_env_state: StateWithMetrics) -> Transition:
    """Convert a continuing rollout trajectory into replay transitions."""
    next_obs = jnp.concatenate([trajectory.obs[:, 1:], final_env_state.obs[:, None]], axis=1)
    return Transition(
        obs=trajectory.obs,
        next_obs=next_obs,
        info=trajectory.info,
        actions=trajectory.actions,
        rewards=trajectory.rewards,
        action_masks=trajectory.action_masks,
        network_output=trajectory.network_output,
        dones=trajectory.dones,
        truncations=trajectory.truncations,
        episode_returns=trajectory.episode_returns,
        episode_lengths=trajectory.episode_lengths,
    )
