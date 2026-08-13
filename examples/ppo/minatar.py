"""Train PPO on MinAtar"""

import math
from dataclasses import dataclass

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput, compute_total_updates
from modrax.env.base import StateWithMetrics
from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Config, Shape
from modrax.utils import add_cli


@dataclass(frozen=True)
class MinAtarNetworkConfig(Config):
    encoder_hidden_dims: tuple[int, ...] = (128,)
    encoder_dim: int = 64
    policy_hidden_dims: tuple[int, ...] = (64, 64)
    value_hidden_dims: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class MinAtarConfig(TrainConfig):
    env_cfg: GymnaxConfig = GymnaxConfig(env_name="Asterix-MinAtar")
    network_cfg: MinAtarNetworkConfig = MinAtarNetworkConfig()
    optimizer_cfg: OptimizerConfig = OptimizerConfig(learning_rate=3e-4, gradient_clip=0.5)
    alg_cfg: PPOConfig = PPOConfig(
        total_steps=250_000_000,
        num_gen_steps=128,
        num_minibatches=32,
        num_updates=3,
        num_envs=4096,
    )
    wandb: WandbConfig = WandbConfig(enabled=False, project="modrax")
    eval_interval: int = 100
    seed: int = 0
    save_path: None = None


class MinAtarNetwork(PPONetwork):
    def __init__(
        self, obs_shape: Shape, num_actions: int, cfg: MinAtarNetworkConfig, rngs: nnx.Rngs
    ):
        self.is_recurrent = False

        relu = jax.nn.relu
        self.encoder = MLP(
            math.prod(obs_shape), cfg.encoder_hidden_dims, cfg.encoder_dim, relu, rngs=rngs
        )
        self.policy_head = MLP(
            cfg.encoder_dim, cfg.policy_hidden_dims, num_actions, relu, rngs=rngs
        )
        self.value_head = MLP(cfg.encoder_dim, cfg.value_hidden_dims, 1, relu, rngs=rngs)

    def _flatten_obs(self, obs: Float[Array, "B ..."]) -> Float[Array, "B D"]:
        return obs.reshape(obs.shape[0], -1)

    def _forward(self, x: Float[Array, "... D"]):
        x = self.encoder(x)
        return self.policy_head(x), self.value_head(x)

    def train_forward(self, obs: Float[Array, "B T ..."], *args):
        flat_obs = obs.reshape(*obs.shape[:2], -1)
        policy_logits, value = self._forward(flat_obs)
        return PPONetworkOutput(policy_logits, value, None)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        policy_logits, value = self._forward(self._flatten_obs(env_state.obs))

        action = softmax_policy(policy_logits, env_state.action_mask, key)
        return action, PPONetworkOutput(policy_logits, value, None)


@add_cli
def main(cfg: MinAtarConfig):
    key = jax.random.key(cfg.seed)

    # Init objects
    env = GymnaxEnv(cfg.env_cfg)
    network = MinAtarNetwork(env.obs_shape, env.action_size, cfg.network_cfg, nnx.Rngs(cfg.seed))
    optimizer = Optimizer(cfg.optimizer_cfg, network)
    alg = PPOAlg(env, network, optimizer, cfg.alg_cfg, key=key)

    train(alg, cfg)


if __name__ == "__main__":
    main()
