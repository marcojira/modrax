from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float, Key
from pydantic import ConfigDict

from modrax.alg.base import Alg, AlgConfig
from modrax.env.base import Env
from modrax.network.base import Network, NetworkConfig
from modrax.network.block.base import Block, BlockConfig, RecurrentBlock
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.rollout import RolloutData, Trajectory
from modrax.rollout.recurrent_rollout import (
    jit_recurrent_rollout,
    recurrent_rollout,
)
from modrax.rollout.rollout import jit_rollout, rollout
from modrax.types import Shape
from modrax.utils import compute_training_metrics, make_trajectory_minibatches, update_network


class PPONetworkConfig(NetworkConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    encoders: dict[str, tuple[type[Block], BlockConfig, Shape | None]]
    encoder_dim: int

    trunk: tuple[type, BlockConfig]  # Block (feedforward) or RecurrentBlock (recurrent)
    trunk_dim: int

    heads: dict[str, tuple[type[Block], BlockConfig, int | None]]


class PPONetwork(Network):
    def __init__(self, obs_shape: Shape, num_actions: int, cfg: PPONetworkConfig, rngs: nnx.Rngs):
        # Build encoders
        self.encoders = {
            name: block_cls(
                input_shape=obs_shape if shape is None else shape,
                output_dim=cfg.encoder_dim,
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config, shape) in cfg.encoders.items()
        }

        # Build trunk (feedforward or recurrent)
        trunk_cls, trunk_config = cfg.trunk
        trunk_input_dim = len(self.encoders) * cfg.encoder_dim
        self.trunk = trunk_cls(
            input_shape=trunk_input_dim,
            output_dim=cfg.trunk_dim,
            config=trunk_config,
            rngs=rngs,
        )

        # Build heads
        self.heads = {
            name: block_cls(
                input_shape=cfg.trunk_dim,
                output_dim=num_actions if output_dim is None else output_dim,
                config=block_config,
                rngs=rngs,
            )
            for name, (block_cls, block_config, output_dim) in cfg.heads.items()
        }

    @property
    def is_recurrent(self):
        return isinstance(self.trunk, RecurrentBlock)

    def init_recurrent_state(self, batch_size: int):
        return self.trunk.init_recurrent_state(batch_size)

    def reset_recurrent_state(self, recurrent_state, dones):
        return self.trunk.reset_recurrent_state(recurrent_state, dones)

    def __call__(self, inputs: dict[str, Float[Array, "B ..."]], recurrent_state=None):
        encoded = [self.encoders[name](inputs[name]) for name in sorted(self.encoders.keys())]
        features = jnp.concatenate(encoded, axis=-1)

        if self.is_recurrent:
            features = features[:, None, :]  # Add time dimension
            features, new_state = self.trunk(features, recurrent_state)
            features = features.squeeze(1)
            outputs = {name: head(features) for name, head in self.heads.items()}
            return outputs, new_state

        features = self.trunk(features)
        return {name: head(features) for name, head in self.heads.items()}

    def train_forward(self, data):
        obs = data.trajectory.obs
        B, T = obs.shape[0], obs.shape[1]

        # Encode: flatten [B, T, ...] -> [B*T, ...], encode
        encoded = []
        for name in sorted(self.encoders.keys()):
            x = getattr(data.trajectory, name)
            x = self.encoders[name](x.reshape(-1, *x.shape[2:]))
            encoded.append(x)
        features = jnp.concatenate(encoded, axis=-1)

        if self.is_recurrent:
            # Unflatten for recurrent processing
            features = features.reshape(B, T, -1)
            features = self.trunk.train_forward(
                features, data.recurrent_output, data.init_recurrent_state
            )
            features = features.reshape(-1, *features.shape[2:])
        else:
            features = self.trunk(features)

        outputs = {name: head(features) for name, head in self.heads.items()}
        return jax.tree.map(lambda x: x.reshape(B, T, *x.shape[1:]), outputs)


class PPOConfig(AlgConfig):
    model_config = {"arbitrary_types_allowed": True}

    network_cfg: PPONetworkConfig
    optimizer_cfg: OptimizerConfig = OptimizerConfig()

    num_envs: int

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


@struct.dataclass
class PPOState:
    network_state: Any
    env_state: Any
    recurrent_state: Any


class PPOAlg(Alg):
    def __init__(self, env: Env, alg_cfg: PPOConfig, key: Key[Array, ""], jit: bool = False):
        self.env = env
        self.cfg = alg_cfg
        self.jit = jit

        key, network_key, reset_key = jax.random.split(key, 3)

        # Create network and optimizer
        network = PPONetwork(
            env.obs_shape, env.action_size, alg_cfg.network_cfg, nnx.Rngs(network_key)
        )
        network.eval()
        optimizer = Optimizer(alg_cfg.optimizer_cfg, network)

        # Initialize environment
        env_state = env.reset(jax.random.split(reset_key, alg_cfg.num_envs))

        # Initialize recurrent state and select rollout function
        recurrent_state = None
        if network.is_recurrent:
            recurrent_state = network.init_recurrent_state(alg_cfg.num_envs)
            self.rollout_fn = jit_recurrent_rollout if jit else recurrent_rollout
        else:
            self.rollout_fn = jit_rollout if jit else rollout

        self.env_steps_per_epoch = alg_cfg.num_envs * alg_cfg.num_gen_steps

        self.state = PPOState(
            network_state=nnx.split((network, optimizer)),
            env_state=env_state,
            recurrent_state=recurrent_state,
        )
        self.loop = nnx.jit(self._loop) if jit else self._loop

    def _loop(self, state: PPOState, key: Key[Array, ""]):
        network, optimizer = nnx.merge(*state.network_state)
        rollout_key, update_key = jax.random.split(key)

        # Generate data
        env_state, recurrent_state, data = self.rollout_fn(
            network,
            softmax_policy,
            self.env.step,
            state.env_state,
            state.recurrent_state,
            self.cfg.num_gen_steps,
            rollout_key,
        )

        # Run multiple epochs of updates
        for _ in range(self.cfg.num_epochs):
            update_key, epoch_key = jax.random.split(update_key)
            _, infos = update(network, optimizer, data, self.cfg, epoch_key)

        # Compute training metrics from trajectory and combine with loss info from last epoch
        metrics = compute_training_metrics(data.trajectory)
        metrics.update(infos)

        new_state = PPOState(
            network_state=nnx.split((network, optimizer)),
            env_state=env_state,
            recurrent_state=recurrent_state,
        )

        return new_state, metrics

    def get_network(self):
        network, _ = nnx.merge(*self.state.network_state)
        return network

    def __call__(self, key: Key[Array, ""]):
        self.state, metrics = self.loop(self.state, key)
        return metrics
