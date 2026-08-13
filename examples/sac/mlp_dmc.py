import math
import os

from modrax.env.base import StateWithMetrics
from modrax.utils import add_cli

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.sac import (
    Actor,
    Critic,
    LogAlpha,
    SACAlg,
    SACConfig,
    SACNetwork,
    SACNetworkConfig,
    SACOptimizer,
    SACOptimizerConfig,
)
from modrax.env.mujoco_playground import MuJoCoPlaygroundConfig, MuJoCoPlaygroundEnv
from modrax.network.mlp import MLP
from modrax.network.running_norm import RunningNorm
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Shape


class MLPActor(Actor):
    def __init__(self, obs_shape: Shape, action_size: int, cfg: SACNetworkConfig, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(math.prod(obs_shape), 256, rngs=rngs)
        self.fc2 = nnx.Linear(256, 256, rngs=rngs)
        self.fc_mean = nnx.Linear(256, action_size, rngs=rngs)
        self.fc_std = nnx.Linear(256, action_size, rngs=rngs)
        self.min_std = cfg.min_std

    def __call__(self, x: Float[Array, "B D"]):
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        mean = self.fc_mean(x)
        std = jax.nn.softplus(self.fc_std(x)) + self.min_std
        return mean, std

    def get_action(self, x: Float[Array, "B D"], key: Key[Array, ""]):
        mean, std = self(x)

        # Reparameterization trick (pre-tanh)
        x_t = mean + std * jax.random.normal(key, mean.shape)

        # Log prob with numerically stable tanh Jacobian correction
        log_prob = -0.5 * ((x_t - mean) / std) ** 2 - 0.5 * jnp.log(2 * jnp.pi) - jnp.log(std)
        log_det_jacobian = 2.0 * (jnp.log(2.0) - x_t - jax.nn.softplus(-2.0 * x_t))
        log_prob = log_prob - log_det_jacobian
        log_prob = jnp.sum(log_prob, axis=-1)

        action = jnp.tanh(x_t)
        return action, log_prob


class MLPCritic(Critic):
    def __init__(self, obs_shape: Shape, action_size: int, cfg: SACNetworkConfig, rngs: nnx.Rngs):
        flat_input_size = math.prod(obs_shape) + action_size

        self.soft_q_1 = MLP(
            flat_input_size, (256, 256), 1, activation_fn=jax.nn.relu, rngs=rngs, layer_norm=True
        )
        self.soft_q_2 = MLP(
            flat_input_size, (256, 256), 1, activation_fn=jax.nn.relu, rngs=rngs, layer_norm=True
        )

    def get_qs(self, obs: Float[Array, "B D"], action: Float[Array, "B A"]):
        x = jnp.concat([obs, action], axis=-1)
        return self.soft_q_1(x).squeeze(-1), self.soft_q_2(x).squeeze(-1)


class MuJoCoSACNetwork(SACNetwork):
    def __init__(self, obs_shape: Shape, action_size: int, cfg: SACNetworkConfig, rngs: nnx.Rngs):
        self.critic = MLPCritic(obs_shape, action_size, cfg, rngs)
        self.critic_target = nnx.clone(self.critic)
        self.actor = MLPActor(obs_shape, action_size, cfg, rngs)
        self.log_alpha = LogAlpha(math.log(cfg.init_alpha))

        self.running_norm = RunningNorm(obs_shape) if cfg.running_norm else lambda x: x

    def policy(self, env_state: StateWithMetrics, key):
        obs = self.running_norm(env_state.obs)
        return self.actor.get_action(obs, key)

    def eval_policy(self, env_state: StateWithMetrics, key):
        obs = self.running_norm(env_state.obs)
        mean, _ = self.actor(obs)
        return jnp.tanh(mean), None


@add_cli
def main(cfg: MuJoCoPlaygroundConfig):
    env_config = MuJoCoPlaygroundConfig(env_name="CartpoleBalance")
    network_config = SACNetworkConfig(running_norm=True)
    optimizer_config = SACOptimizerConfig()
    alg_config = SACConfig()
    train_config = TrainConfig(
        seed=0,
        env_cfg=env_config,
        network_cfg=network_config,
        optimizer_cfg=optimizer_config,
        alg_cfg=alg_config,
        save_path=None,
        wandb=WandbConfig(enabled=True),
        save_gif_wandb=True,
        eval_interval=5,
    )

    env = MuJoCoPlaygroundEnv(env_config)
    network = MuJoCoSACNetwork(
        env.obs_shape, env.action_size, network_config, nnx.Rngs(train_config.seed)
    )
    optimizer = SACOptimizer(network.actor, network.critic, network.log_alpha, optimizer_config)
    alg = SACAlg(env, network, optimizer, alg_config, jax.random.key(train_config.seed))

    trained_network = train(alg, train_config)
    return trained_network


if __name__ == "__main__":
    main()
