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
from modrax.policy import argmax_policy
from modrax.utils import compute_training_metrics


class RolloutReturnConfig(EvalConfig):
    num_envs: int = 512
    num_gen_steps: int = 1000
    rollout_fn: Callable
    policy_fn: Callable


def rollout_return(
    network: Network,
    env: Env,
    rollout_fn: RolloutFn,
    config: RolloutReturnConfig,
    key: Key[Array, ""],
) -> dict[str, float]:
    env_state = env.reset(jax.random.split(key, config.num_envs))

    recurrent_state = None
    if isinstance(network, RecurrentNetwork):
        recurrent_state = network.init_recurrent_state(config.num_envs)

    env_state, recurrent_state, data = config.rollout_fn(
        network,
        config.policy_fn,
        env.step,
        env_state,
        recurrent_state,
        config.num_gen_steps,
        key,
    )

    metrics = compute_training_metrics(data.trajectory)
    return metrics
