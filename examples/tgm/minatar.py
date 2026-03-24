from dataclasses import dataclass

import jax
from flax import nnx

from modrax.alg.tgm import (
    TGMAlg,
    TGMConfig,
    TGMNetwork,
    TGMNetworkOutput,
    compute_total_updates,
)
from modrax.env.base import StateWithMetrics
from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import epsilon_softmax_policy
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Config


@dataclass
class MinatarNetworkConfig(Config):
    norm_type: str = "layer_norm"  # "layer_norm" | "batch_norm" | "none"
    norm_input: bool = False


@dataclass
class MinatarConfig(TrainConfig):
    env_cfg: GymnaxConfig = GymnaxConfig(env_name="Asterix-MinAtar")
    network_cfg: MinatarNetworkConfig = MinatarNetworkConfig(norm_type="layer_norm")
    optimizer_cfg: OptimizerConfig = OptimizerConfig(
        optimizer_type="adam", learning_rate=1e-3, lr_decay=True, gradient_clip=10
    )
    alg_cfg: TGMConfig = TGMConfig(num_envs=128, num_timesteps=32, num_minibatches=32)
    wandb: WandbConfig = WandbConfig(enabled=False, project="modrax")
    eval_interval: int = 250
    seed: int = 0
    save_gif_wandb: bool = True


class MinatarNetwork(TGMNetwork):
    is_recurrent = False

    def __init__(
        self,
        obs_shape,
        action_size: int,
        cfg: MinatarNetworkConfig,
        alpha: float,
        q: float,
        omega: float,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.eps = 1.0
        self.alpha = alpha
        self.q = q
        self.omega = omega
        h, w, in_channels = obs_shape

        self.conv = nnx.Conv(
            in_channels,
            16,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs,
        )
        self.dense = nnx.Linear(
            (h - 2) * (w - 2) * 16,
            128,
            kernel_init=nnx.initializers.he_normal(),
            rngs=rngs,
        )
        self.q_head = nnx.Linear(128, action_size, rngs=rngs)
        self.v_head = nnx.Linear(128, 1, rngs=rngs)

        self.input_norm = nnx.BatchNorm(in_channels, rngs=rngs) if cfg.norm_input else None

        if cfg.norm_type == "layer_norm":
            self.norm1 = nnx.LayerNorm(16, rngs=rngs)
            self.norm2 = nnx.LayerNorm(128, rngs=rngs)
        elif cfg.norm_type == "batch_norm":
            self.norm1 = nnx.BatchNorm(16, rngs=rngs)
            self.norm2 = nnx.BatchNorm(128, rngs=rngs)
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
            x = x / 255.0

        x = jax.nn.relu(self._apply_norm(self.conv(x), self.norm1, train))
        x = x.reshape(x.shape[0], -1)
        x = jax.nn.relu(self._apply_norm(self.dense(x), self.norm2, train))
        q_values = self.q_head(x)
        value = self.v_head(x).squeeze(-1)
        return q_values, value

    def policy(self, env_state: StateWithMetrics, key):
        q_values, value = self.__call__(env_state.obs)

        action = epsilon_softmax_policy(q_values, env_state.action_mask, key, self.eps)
        return action, TGMNetworkOutput(q_values, value, None)

    def train_forward(self, obs):
        B, T = obs.shape[:2]
        flat_obs = obs.reshape(B * T, *obs.shape[2:])
        q_values, value = self.__call__(flat_obs)
        return q_values.reshape(B, T, -1), value.reshape(B, T)


def main():
    cfg = MinatarConfig()
    key = jax.random.key(cfg.seed)

    # Init objects
    env = GymnaxEnv(cfg.env_cfg)
    network = MinatarNetwork(
        env.obs_shape,
        env.action_size,
        cfg.network_cfg,
        cfg.alg_cfg.alpha,
        cfg.alg_cfg.q,
        cfg.alg_cfg.omega,
        nnx.Rngs(cfg.seed),
    )
    optimizer = Optimizer(cfg.optimizer_cfg, network, compute_total_updates(cfg.alg_cfg))
    alg = TGMAlg(env, network, optimizer, cfg.alg_cfg, key=key)

    train(env, network, optimizer, alg, cfg)


if __name__ == "__main__":
    main()
