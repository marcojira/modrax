from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from modrax.alg.base import OptimizerConfig
from modrax.alg.pqn import PQNAlg, PQNConfig, PQNNetwork, PQNNetworkOutput
from modrax.cli import add_cli
from modrax.env.base import StateWithMetrics
from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
from modrax.network import NetworkConfig
from modrax.policy import epsilon_greedy_policy
from modrax.training import TrainConfig, WandbConfig, train


@dataclass(frozen=True)
class MinAtarNetworkConfig(NetworkConfig):
    conv_features: int = 16
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    hidden_dim: int = 128
    norm_type: str = "layer_norm"  # "layer_norm" | "batch_norm" | "none"
    norm_input: bool = False


@dataclass(frozen=True)
class MinAtarConfig(TrainConfig):
    env_cfg: GymnaxConfig = GymnaxConfig(env_name="Asterix-MinAtar")
    network_cfg: MinAtarNetworkConfig = MinAtarNetworkConfig(norm_type="layer_norm")
    alg_cfg: PQNConfig = PQNConfig(
        optimizer_cfg=OptimizerConfig(
            optimizer_type="adamw", learning_rate=5e-4, lr_decay=True, gradient_clip=10
        )
    )
    wandb: WandbConfig = WandbConfig(enabled=False, project="modrax")
    eval_interval: int = 250
    seed: int = 0
    save_gif_wandb: bool = True


class MinAtarNetwork(PQNNetwork):
    def __init__(self, obs_shape, action_size: int, cfg: MinAtarNetworkConfig, rngs: nnx.Rngs):
        self.is_recurrent = False
        self.cfg = cfg
        self.eps = nnx.Variable(jnp.array(1.0))
        h, w, in_channels = obs_shape
        conv_h = (h - cfg.kernel_size[0]) // cfg.strides[0] + 1
        conv_w = (w - cfg.kernel_size[1]) // cfg.strides[1] + 1

        self.conv = nnx.Conv(
            in_channels,
            cfg.conv_features,
            kernel_size=cfg.kernel_size,
            strides=cfg.strides,
            padding="VALID",
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs,
        )
        self.dense = nnx.Linear(
            conv_h * conv_w * cfg.conv_features,
            cfg.hidden_dim,
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs,
        )
        self.output = nnx.Linear(cfg.hidden_dim, action_size, rngs=rngs)

        self.input_norm = nnx.BatchNorm(in_channels, rngs=rngs) if cfg.norm_input else None

        if cfg.norm_type == "layer_norm":
            self.norm1 = nnx.LayerNorm(cfg.conv_features, rngs=rngs)
            self.norm2 = nnx.LayerNorm(cfg.hidden_dim, rngs=rngs)
        elif cfg.norm_type == "batch_norm":
            self.norm1 = nnx.BatchNorm(cfg.conv_features, rngs=rngs)
            self.norm2 = nnx.BatchNorm(cfg.hidden_dim, rngs=rngs)
        else:
            self.norm1 = self.norm2 = None

    def _apply_norm(self, x, norm, train: bool):
        if norm is None:
            return x
        if isinstance(norm, nnx.BatchNorm):
            return norm(x, use_running_average=not train)
        return norm(x)

    def __call__(self, x, train: bool = True):
        if self.input_norm is not None:
            x = self.input_norm(x, use_running_average=not train)
        else:
            x = x

        x = jax.nn.relu(self._apply_norm(self.conv(x), self.norm1, train))
        x = x.reshape(x.shape[0], -1)
        x = jax.nn.relu(self._apply_norm(self.dense(x), self.norm2, train))
        return self.output(x)

    def policy(self, env_state: StateWithMetrics, key):
        q_values = self.__call__(env_state.obs)
        action = epsilon_greedy_policy(q_values, env_state.action_mask, key, self.eps)
        return action, PQNNetworkOutput(q_values, None)

    def eval_policy(self, env_state: StateWithMetrics, key):
        q_values = self.__call__(env_state.obs, train=False)
        action = epsilon_greedy_policy(q_values, env_state.action_mask, key, 0.0)
        return action, PQNNetworkOutput(q_values, None)

    def train_forward(self, obs):
        return self.__call__(obs)


@add_cli
def main(cfg: MinAtarConfig):
    key = jax.random.key(cfg.seed)
    network_key, alg_key, train_key = jax.random.split(key, 3)

    # Init objects
    env = GymnaxEnv(cfg.env_cfg)
    network = MinAtarNetwork(env.obs_shape, env.action_size, cfg.network_cfg, nnx.Rngs(network_key))
    alg = PQNAlg(env, network, cfg.alg_cfg, key=alg_key)

    train(alg, cfg, key=train_key)


if __name__ == "__main__":
    main()
