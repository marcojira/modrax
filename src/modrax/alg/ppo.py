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


class PPOConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    minibatch_size: int = 4096
    num_epochs: int = 3


""" HELPERS """


def compute_gae_advantages(
    trajectory: Trajectory, last_val: Float[Array, " B"], config: PPOConfig
) -> tuple[Float[Array, "B T"], Float[Array, "B T"]]:
    def backwards_fn(gae_and_next_val, transition):
        gae, next_val = gae_and_next_val
        done, val, reward = transition

        delta = reward + config.gamma * next_val * (1 - done) - val
        gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae

        return (gae, val), gae

    values = trajectory.network_output["value"].squeeze(-1)

    # Transpose to (T, B) for scan, then transpose back
    transitions = (trajectory.dones.T, values.T, trajectory.rewards.T)
    _, advantages = jax.lax.scan(
        backwards_fn,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )

    advantages = advantages.T
    returns = advantages + values
    return advantages, returns


def ppo_loss(network: Network, minibatch, config: PPOConfig):
    out = network.train_forward(minibatch["data"])
    values, action_logits = out["value"], out["policy"]
    values = values.squeeze(-1)

    batch = minibatch["data"].trajectory
    old_values = batch.network_output["value"].squeeze(-1)

    # Apply action mask to logits
    action_mask = batch.action_masks
    masked_logits = jnp.where(action_mask, action_logits, -jnp.inf)
    masked_prev_logits = jnp.where(action_mask, batch.network_output["policy"], -jnp.inf)

    prev_log_probs = jax.nn.log_softmax(masked_prev_logits, axis=-1)  # type: ignore
    log_probs = jax.nn.log_softmax(masked_logits, axis=-1)  # type: ignore

    # Get log probs for taken actions
    actions = batch.actions[..., None]
    current_log_probs = jnp.take_along_axis(log_probs, actions, axis=-1).squeeze(-1)
    old_log_probs = jnp.take_along_axis(prev_log_probs, actions, axis=-1).squeeze(-1)

    # Compute entropy (only over valid actions)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * jnp.where(action_mask, log_probs, 0.0), axis=-1)

    # Normalize advantages
    advantages = (minibatch["advantage"] - minibatch["advantage"].mean()) / (
        minibatch["advantage"].std() + 1e-8
    )

    # Actor loss with clipping
    prob_ratio = jnp.exp(current_log_probs - old_log_probs)
    clipped_ratio = prob_ratio.clip(1.0 - config.clip_eps, 1.0 + config.clip_eps)
    actor_loss = -jnp.minimum(prob_ratio * advantages, clipped_ratio * advantages)

    # Critic loss with clipping
    value_pred_clipped = old_values + (values - old_values).clip(-config.clip_eps, config.clip_eps)
    value_losses = jnp.square(values - minibatch["return"])
    value_losses_clipped = jnp.square(value_pred_clipped - minibatch["return"])
    critic_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped)

    # Total loss
    total_loss = (
        actor_loss.mean()
        + config.value_coeff * critic_loss.mean()
        - config.entropy_coeff * entropy.mean()
    )

    info = {
        "actor_loss": actor_loss.mean(),
        "critic_loss": critic_loss.mean(),
        "entropy": entropy.mean(),
        "total_loss": total_loss,
    }

    return total_loss, info


def update(
    network: Network,
    optimizer: Optimizer,
    data: RolloutData,
    config: PPOConfig,
    key: Key[Array, ""],
) -> tuple[Float[Array, ""], dict]:
    # Compute last value for GAE bootstrapping
    advantages, returns = compute_gae_advantages(
        data.trajectory,
        data.final_out["value"].squeeze(-1),
        config,
    )

    # Add advantages and returns to trajectory
    all_data = {
        "data": data,
        "advantage": advantages,
        "return": returns,
    }

    minibatches = make_trajectory_minibatches(all_data, key, config.minibatch_size)

    loss, infos = update_network(network, optimizer, minibatches, ppo_loss, config)
    return loss, infos


""" ALGORITHM """


class PPOAlg(Alg):
    def __init__(
        self,
        env_state: EnvState,
        recurrent_state: RecurrentState | None,
        env: Env,
        alg_config: PPOConfig,
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
