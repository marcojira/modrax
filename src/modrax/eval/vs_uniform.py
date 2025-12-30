"""Evaluate a network against a uniform (random) opponent in two-player games."""

from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Key

from modrax.env.base import Env, EnvConfig
from modrax.eval.base import EvalConfig
from modrax.network.base import Network
from modrax.network.recurrent_network import RecurrentNetwork
from modrax.policy import softmax_policy
from modrax.rollout.base import RolloutData


def compute_outcome_stats(outcomes: Array) -> dict[str, float]:
    """Compute win/draw/loss rates from outcome array."""
    total = len(outcomes)
    return {
        "win_rate": float(jnp.sum(outcomes > 0) / total),
        "draw_rate": float(jnp.sum(outcomes == 0) / total),
        "loss_rate": float(jnp.sum(outcomes < 0) / total),
    }


class VsUniformConfig(EvalConfig):
    env_config: EnvConfig
    num_games: int = 100


def play_games(
    network_1: Network | Callable,
    network_2: Network | Callable,
    env: Env,
    key: Key[Array, ""],
    num_games: int = 100,
) -> Array:
    """Play games between two networks, returning outcomes from network_1's perspective."""

    def step(carry):
        (step_num, dones, outcomes, network_1, network_2, env_state, key) = carry
        obs = env_state.obs
        action_mask = env_state.action_mask

        key, action_key, env_key = jax.random.split(key, 3)

        logits_1 = network_1({"obs": obs})["policy"]
        logits_2 = network_2({"obs": obs})["policy"]

        policy = jnp.where(step_num % 2 == 0, logits_1, logits_2)
        action = softmax_policy(policy, action_mask, action_key)

        # Step environment
        env_keys = jax.random.split(env_key, obs.shape[0])
        step_output, new_env_state = env.step(env_state, action, env_keys)

        # Get reward from first player's perspective
        first_player_reward = jnp.where(step_num % 2 == 0, step_output.reward, -step_output.reward)
        outcomes = jnp.where(dones, outcomes, first_player_reward)
        dones = jnp.logical_or(dones, step_output.done)

        return (step_num + 1, dones, outcomes, network_1, network_2, new_env_state, key)

    key, reset_key = jax.random.split(key)
    env_state = env.reset(jax.random.split(reset_key, num_games))
    init = (
        0,
        jnp.zeros(num_games, dtype=jnp.bool),
        jnp.zeros(num_games),
        network_1,
        network_2,
        env_state,
        key,
    )

    _, _, outcomes, _, _, _, _ = nnx.while_loop(lambda x: jnp.any(~x[1]), step, init)
    return outcomes  # type: ignore


@nnx.jit(static_argnames=("env", "num_games"))
def play_games_both_sides(
    network: Network | Callable,
    opponent: Network | Callable,
    env: Env,
    key: Key[Array, ""],
    num_games: int,
) -> Array:
    """Play games as both P1 and P2, returning combined outcomes from network's perspective."""
    if isinstance(network, RecurrentNetwork):
        network.set_recurrent_state(num_games)
    if isinstance(opponent, RecurrentNetwork):
        opponent.set_recurrent_state(num_games)

    key1, key2 = jax.random.split(key)
    outcomes_as_p1 = play_games(network, opponent, env, key1, num_games)
    outcomes_as_p2 = -play_games(opponent, network, env, key2, num_games)
    return jnp.concatenate([outcomes_as_p1, outcomes_as_p2])


def eval_vs_uniform(
    network: Network,
    env: Env,
    rollout_data: RolloutData,
    config: VsUniformConfig,
    key: Key[Array, ""],
) -> dict[str, float]:
    class UniformNetwork(Network):
        def __init__(self):
            return

        def __call__(self, *args):
            return {"policy": jnp.ones((config.num_games, env.num_actions))}

    outcomes = play_games_both_sides(network, UniformNetwork(), env, key, config.num_games)
    return compute_outcome_stats(outcomes)
