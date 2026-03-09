"""Train PPO with GTrXL on Craftax."""

import math
from dataclasses import dataclass

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput, compute_total_updates
from modrax.env.base import StateWithMetrics
from modrax.env.craftax import CraftaxConfig, CraftaxEnv
from modrax.network.gtrxl import GatedTransformerXL, GTrXLRecurrentState
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Config, Shape


@dataclass
class CraftaxGTrXLNetworkConfig(Config):
    encoder_dim: int = 256
    policy_hidden_dims: tuple[int, ...] = (256, 256)
    value_hidden_dims: tuple[int, ...] = (256, 256)
    num_heads: int = 8
    num_layers: int = 2
    gating: bool = True
    gating_bias: float = 2.0
    segment_len: int = 64
    rollout_memory_len: int = 128
    cached_train: bool = True


@dataclass
class CraftaxGTrXLPPOConfig(TrainConfig):
    env_cfg: CraftaxConfig = CraftaxConfig(
        env_name="Craftax-Symbolic-v1",
        optimistic_reset=True,
    )
    network_cfg: CraftaxGTrXLNetworkConfig = CraftaxGTrXLNetworkConfig()
    optimizer_cfg: OptimizerConfig = OptimizerConfig(
        learning_rate=2e-4, gradient_clip=1.0, lr_decay=True
    )
    alg_cfg: PPOConfig = PPOConfig(
        total_steps=1_000_000_000,
        num_gen_steps=129,  # + 1 since we use 1 for bootstrapping
        num_minibatches=8,
        gamma=0.999,
        gae_lambda=0.8,
        entropy_coeff=0.002,
        num_updates=4,
        num_envs=1024,
    )
    wandb: WandbConfig = WandbConfig(enabled=True, project="modrax")
    eval_interval: int = 100
    seed: int = 0
    save_gif_wandb: bool = True
    num_gif_trajectories: int = 2


class CraftaxGTrXLNetwork(PPONetwork):
    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        num_envs: int,
        cfg: CraftaxGTrXLNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.is_recurrent = True

        relu = jax.nn.relu
        self.encoder = nnx.Linear(math.prod(obs_shape), cfg.encoder_dim, rngs=rngs)
        self.gtrxl = GatedTransformerXL(
            input_dim=cfg.encoder_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            rollout_memory_len=cfg.rollout_memory_len,
            segment_len=cfg.segment_len,
            rngs=rngs,
            gating=cfg.gating,
            gating_bias=cfg.gating_bias,
            cached_train=cfg.cached_train,
        )
        self.gtrxl.initialize_carry(num_envs)
        self.policy_head = MLP(
            cfg.encoder_dim, cfg.policy_hidden_dims, num_actions, relu, rngs=rngs
        )
        self.value_head = MLP(cfg.encoder_dim, cfg.value_hidden_dims, 1, relu, rngs=rngs)

    def train_forward(
        self,
        obs: Float[Array, "B T ..."],
        dones: Float[Array, "B T"],
        init_carry: GTrXLRecurrentState,
        saved_carry: GTrXLRecurrentState,
    ):
        encoded = self.encoder(obs.reshape(*obs.shape[:2], -1))
        encoded = self.gtrxl.train_forward(encoded, init_carry, saved_carry)

        policy_logits = self.policy_head(encoded)
        value = self.value_head(encoded)
        return PPONetworkOutput(policy_logits, value, None)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self.encoder(obs)
        carry, out = self.gtrxl(encoded)

        policy_logits, value = self.policy_head(out), self.value_head(out)
        action = softmax_policy(policy_logits, env_state.action_mask, key)
        return action, PPONetworkOutput(policy_logits, value, carry)

    def reset(self, done: Float[Array, " B"]):
        self.gtrxl.reset(done)

    def get_carry(self):
        return self.gtrxl.carry.value


def main(cfg):
    key = jax.random.key(cfg.seed)

    # Init objects
    env = CraftaxEnv(cfg.env_cfg)
    network = CraftaxGTrXLNetwork(
        env.obs_shape, env.action_size, cfg.alg_cfg.num_envs, cfg.network_cfg, nnx.Rngs(cfg.seed)
    )
    optimizer = Optimizer(
        cfg.optimizer_cfg, network, total_num_updates=compute_total_updates(cfg.alg_cfg)
    )
    alg = PPOAlg(env, network, optimizer, cfg.alg_cfg, key=key)

    train(env, network, optimizer, alg, cfg)


if __name__ == "__main__":
    cfg = CraftaxGTrXLPPOConfig()
    main(cfg)
