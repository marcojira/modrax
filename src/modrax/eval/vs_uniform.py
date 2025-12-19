"""Evaluate a network against a uniform (random) opponent in two-player games."""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Key

from modrax.env.base import Env, EnvConfig
from modrax.eval.base import EvalConfig
from modrax.network.base import Network
from modrax.policy import softmax_policy
from modrax.rollout.base import RolloutData


class VsUniformConfig(EvalConfig):
    env_config: EnvConfig
    num_games: int = 100


def play_games(
    player_1: Callable,
    player_2: Callable,
    env: Env,
    key: Key[Array, ""],
    num_games: int = 100,
) -> Array:
    def step(carry):
        step, dones, outcomes, env_state, key = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        key, action_key, env_key = jax.random.split(key, 3)

        # Take action
        action = jax.lax.select(
            step % 2 == 0,
            player_1(obs, action_mask, action_key),
            player_2(obs, action_mask, action_key),
        )

        # Step
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = env.step(env_state, action, env_keys)

        # Get reward from first player's perspective
        first_player_reward = jnp.where(step % 2 == 0, step_output.reward, -step_output.reward)
        outcomes = jnp.where(
            dones, outcomes, first_player_reward
        )  # Only record reward if not already done
        dones = jnp.logical_or(dones, step_output.done)

        return (step + 1, dones, outcomes, new_env_state, key)

    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, num_games))
    init = (
        0,  # step
        jnp.zeros(num_games, dtype=jnp.bool),  # dones
        jnp.zeros(num_games),  # outcomes
        env_state,
        key,
    )

    # Keep running while any is not done
    final_step, final_dones, outcomes, _, _ = jax.lax.while_loop(
        lambda x: jnp.any(~x[1]), step, init
    )
    return outcomes  # type: ignore


def eval_vs_uniform(
    network: Network,
    env: Env,
    rollout_data: RolloutData,
    config: VsUniformConfig,
    key: Key[Array, ""],
) -> dict[str, float]:
    def network_player(obs, action_mask, action_key):
        out = network({"obs": obs})
        logits = out["policy"]
        action = softmax_policy(logits, action_mask, action_key)
        return action

    def uniform_player(obs, action_mask, action_key):
        logits = jnp.zeros(action_mask.shape)
        action = softmax_policy(logits, action_mask, action_key)
        return action

    num_games = config.num_games
    key1, key2 = jax.random.split(key)
    # outcomes = play_games(uniform_player, uniform_player, env, key1, num_games)
    outcomes_as_p1 = play_games(network_player, uniform_player, env, key1, num_games)
    outcomes_as_p2 = -play_games(uniform_player, network_player, env, key2, num_games)
    outcomes = jnp.concatenate([outcomes_as_p1, outcomes_as_p2])

    total_games = 2 * num_games
    wins = jnp.sum(outcomes > 0)
    draws = jnp.sum(outcomes == 0)
    losses = jnp.sum(outcomes < 0)

    return {
        "win_rate": float(wins / total_games),
        "draw_rate": float(draws / total_games),
        "loss_rate": float(losses / total_games),
    }
