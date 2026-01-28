from typing import Callable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.base import Alg, AlgConfig
from modrax.env.base import Env, EnvState
from modrax.network.base import Network
from modrax.network.block.base import RecurrentState
from modrax.optimizer import Optimizer
from modrax.policy import softmax_policy
from modrax.rollout import RolloutData, Trajectory
from modrax.rollout.recurrent_rollout import jit_recurrent_rollout, recurrent_rollout
from modrax.rollout.rollout import jit_rollout, rollout
from modrax.update.base import make_trajectory_minibatches, update_network
from modrax.utils import compute_training_metrics


class GumbelAZConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    minibatch_size: int = 32
    sigma_fn: Callable[[Float[Array, "B T A"]], Float[Array, "B T A"]] = lambda x: 5 * x
    loss: str = "KL"
    num_epochs: int = 3


""" HELPERS """


def compute_value_target(trajectory: Trajectory) -> Float[Array, "B T"]:
    # TODO: This only works for terminal rewards ATM
    def backwards_fn(carry, transition):
        done, reward = transition

        # If done, target is reward, otherwise bring back next value
        value_target = done * reward + (1 - done) * (carry)

        # Inverse value target as carry since previous state is opponent and these are zero-sum
        return -value_target, value_target

    transitions = (trajectory.dones.T, trajectory.rewards.T)

    # Bootstrap with value of network at final step initially
    final_val = trajectory.network_output["value"][:, -1].squeeze(-1)

    _, value_target = jax.lax.scan(backwards_fn, final_val, transitions, reverse=True)
    return value_target.T


def rescale_q_values(q_values: Float[Array, "B T A"], eps: float = 1e-8) -> Float[Array, "B T A"]:
    min_val = jnp.min(q_values, axis=-1, keepdims=True)
    max_val = jnp.max(q_values, axis=-1, keepdims=True)
    return (q_values - min_val) / jnp.maximum(max_val - min_val, eps)


def compute_policy_target(data: RolloutData, config: GumbelAZConfig):
    # TODO Add final target
    traj = data.trajectory
    B, T, num_actions = traj.network_output["policy"].shape

    # Compute Q-value
    # Since this is the value for the opponent, we negate it to get our value (zero-sum game)
    next_val = jnp.roll(traj.network_output["value"].squeeze(-1), shift=-1, axis=-1)
    q_value = jnp.where(traj.dones, traj.rewards, -next_val)  # ??? why negative

    # Create completed Q-values
    v_mix = (traj.network_output["value"].squeeze(-1) + q_value) / 2
    completed_q = jnp.tile(v_mix[..., None], (1, 1, num_actions))
    completed_q = completed_q.at[jnp.arange(B)[:, None], jnp.arange(T)[None, :], traj.actions].set(
        q_value
    )
    completed_q = rescale_q_values(completed_q)

    # Create logits
    logits = traj.network_output["policy"] + config.sigma_fn(completed_q)
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(traj.action_masks, logits, -jnp.inf)

    return jax.nn.softmax(logits, axis=-1)


def gumbel_az_loss(network: Network, minibatch, config: GumbelAZConfig):
    out = network.train_forward(minibatch["data"])

    action_masks = minibatch["data"].trajectory.action_masks

    if config.loss == "KL":
        log_policy = jax.nn.log_softmax(out["policy"], axis=-1)
        policy_loss = jnp.where(
            minibatch["policy_target"] > 1e-8,
            minibatch["policy_target"] * (jnp.log(minibatch["policy_target"]) - log_policy),
            0,
        )
    elif config.loss == "CE":
        policy_loss = -jax.nn.log_softmax(out["policy"], axis=-1) * minibatch["policy_target"]
        policy_loss = jnp.where(action_masks, policy_loss, 0)
    policy_loss = jnp.sum(policy_loss, axis=-1)  # type: ignore
    policy_loss = policy_loss.mean()

    value = out["value"].squeeze(-1)
    value_loss = 0.5 * (value - minibatch["value_target"]) ** 2
    value_loss = value_loss.mean()

    total_loss = policy_loss + value_loss
    metrics = {"policy_loss": policy_loss, "value_loss": value_loss}
    return total_loss, metrics


def update(
    network: Network,
    optimizer: Optimizer,
    data: RolloutData,
    config: GumbelAZConfig,
    key: Key[Array, ""],
) -> tuple[Float[Array, ""], dict]:
    value_target = compute_value_target(data.trajectory)
    policy_target = compute_policy_target(data, config)

    all_data = {"data": data, "value_target": value_target, "policy_target": policy_target}

    minibatches = make_trajectory_minibatches(all_data, key, config.minibatch_size)

    loss, infos = update_network(network, optimizer, minibatches, gumbel_az_loss, config)
    return loss, infos


""" ALGORITHM """


class GumbelAZAlg(Alg):
    def __init__(
        self,
        env_state: EnvState,
        recurrent_state: RecurrentState | None,
        env: Env,
        alg_config: GumbelAZConfig,
        jit: bool = False,
    ):
        self.env_state = env_state
        self.recurrent_state = recurrent_state

        self.env = env
        self.config = alg_config
        self.jit = jit

        # Select rollout function based on recurrent state
        if recurrent_state is None:
            self.rollout_fn = jit_rollout if jit else rollout
        else:
            self.rollout_fn = jit_recurrent_rollout if jit else recurrent_rollout

        self.update_fn = nnx.jit(update, static_argnames=["config"]) if jit else update

    def __call__(self, network: Network, optimizer: Optimizer, key: Key[Array, ""]):
        rollout_key, update_key = jax.random.split(key)

        # Generate data
        env_state, recurrent_state, data = self.rollout_fn(
            network,
            softmax_policy,
            self.env.step,
            self.env_state,
            self.recurrent_state,  # type: ignore
            self.config.num_gen_steps,
            rollout_key,
        )
        self.env_state, self.recurrent_state = env_state, recurrent_state

        # Run multiple epochs of updates
        epoch_keys = jax.random.split(update_key, self.config.num_epochs)
        for epoch_key in epoch_keys:
            _, infos = self.update_fn(network, optimizer, data, self.config, epoch_key)

        # Compute training metrics from trajectory and combine with loss info from last epoch
        metrics = compute_training_metrics(data.trajectory)
        metrics.update(infos)

        return network, optimizer, metrics
