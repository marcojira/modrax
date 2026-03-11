"""Train PPO with RNN on Craftax."""

import math
from dataclasses import dataclass

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput, compute_total_updates
from modrax.env.base import StateWithMetrics
from modrax.env.craftax import CraftaxConfig, CraftaxEnv
from modrax.network.mlp import MLP
from modrax.network.rnn import NnxRNN
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Config, Shape


@dataclass(frozen=True)
class CraftaxRNNNetworkConfig(Config):
    encoder_dim: int = 256
    rnn_hidden_dim: int = 256
    cell_type: str = "lstm"
    num_rnn_layers: int = 1
    policy_hidden_dims: tuple[int, ...] = (256, 256)
    value_hidden_dims: tuple[int, ...] = (256, 256)


@dataclass(frozen=True)
class CraftaxRNNPPOConfig(TrainConfig):
    env_cfg: CraftaxConfig = CraftaxConfig(
        env_name="Craftax-Symbolic-v1",
        optimistic_reset=True,
    )
    network_cfg: CraftaxRNNNetworkConfig = CraftaxRNNNetworkConfig()
    optimizer_cfg: OptimizerConfig = OptimizerConfig(
        learning_rate=2e-4, gradient_clip=1.0, lr_decay=True
    )
    alg_cfg: PPOConfig = PPOConfig(
        total_steps=1_000_000_000,
        num_gen_steps=129,
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


class CraftaxRNNNetwork(PPONetwork):
    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        num_envs: int,
        cfg: CraftaxRNNNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.is_recurrent = True

        relu = jax.nn.relu
        self.encoder = nnx.Linear(math.prod(obs_shape), cfg.encoder_dim, rngs=rngs)
        self.rnn = NnxRNN(
            input_dim=cfg.encoder_dim,
            output_dim=cfg.rnn_hidden_dim,
            cell_type=cfg.cell_type,
            num_layers=cfg.num_rnn_layers,
            rngs=rngs,
        )
        self.rnn.initialize_carry(num_envs)
        self.policy_head = MLP(
            cfg.rnn_hidden_dim, cfg.policy_hidden_dims, num_actions, relu, rngs=rngs
        )
        self.value_head = MLP(cfg.rnn_hidden_dim, cfg.value_hidden_dims, 1, relu, rngs=rngs)

    def train_forward(
        self, obs: Float[Array, "B T ..."], dones: Float[Array, "B T"], init_carry, saved_carry
    ):
        encoded = self.encoder(obs.reshape(*obs.shape[:2], -1))
        encoded = self.rnn.train_forward(encoded, dones, init_carry)

        policy_logits = self.policy_head(encoded)
        value = self.value_head(encoded)
        return PPONetworkOutput(policy_logits, value, None)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self.encoder(obs)
        carry, out = self.rnn(encoded)

        policy_logits, value = self.policy_head(out), self.value_head(out)
        action = softmax_policy(policy_logits, env_state.action_mask, key)
        return action, PPONetworkOutput(policy_logits, value, carry)

    def reset(self, done: Float[Array, " B"]):
        self.rnn.reset(done)

    def get_carry(self):
        return self.rnn.carry.value


def main(cfg):
    key = jax.random.key(cfg.seed)

    # Init objects
    env = CraftaxEnv(cfg.env_cfg)
    network = CraftaxRNNNetwork(
        env.obs_shape, env.action_size, cfg.alg_cfg.num_envs, cfg.network_cfg, nnx.Rngs(cfg.seed)
    )
    optimizer = Optimizer(
        cfg.optimizer_cfg, network, total_num_updates=compute_total_updates(cfg.alg_cfg)
    )
    alg = PPOAlg(env, network, optimizer, cfg.alg_cfg, key=key)

    train(env, network, optimizer, alg, cfg)


if __name__ == "__main__":
    cfg = CraftaxRNNPPOConfig()
    main(cfg)
