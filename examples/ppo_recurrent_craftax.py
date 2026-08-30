"""Train PPO with LSTM or GTrXL memory on Craftax.

Run once with ``--architecture lstm`` and once with ``--architecture gtrxl``.
"""

import math
from dataclasses import dataclass
from typing import Literal

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.base import OptimizerConfig
from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput
from modrax.cli import add_cli
from modrax.env.base import StateWithMetrics
from modrax.env.craftax import CraftaxConfig, CraftaxEnv
from modrax.network import NetworkConfig
from modrax.network.gtrxl import GatedTransformerXL
from modrax.network.rnn import NnxRNN
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train


@dataclass(frozen=True)
class LSTMConfig(NetworkConfig):
    hidden_dim: int = 512
    num_layers: int = 1


@dataclass(frozen=True)
class GTrXLConfig(NetworkConfig):
    hidden_dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    segment_len: int = 64
    memory_len: int = 128


@dataclass(frozen=True)
class CraftaxNetworkConfig(NetworkConfig):
    architecture: Literal["lstm", "gtrxl"] = "lstm"
    lstm: LSTMConfig = LSTMConfig()
    gtrxl: GTrXLConfig = GTrXLConfig()


@dataclass(frozen=True)
class CraftaxRecurrentPPOConfig(TrainConfig):
    env_cfg: CraftaxConfig = CraftaxConfig(
        env_name="Craftax-Symbolic-v1",
        optimistic_reset=True,
    )
    network_cfg: CraftaxNetworkConfig = CraftaxNetworkConfig()
    alg_cfg: PPOConfig = PPOConfig(
        optimizer_cfg=OptimizerConfig(
            learning_rate=2e-4,
            gradient_clip=1.0,
            lr_decay=True,
        ),
        total_steps=1_000_000_000,
        num_gen_steps=128,
        num_minibatches=8,
        gamma=0.999,
        gae_lambda=0.8,
        entropy_coeff=0.002,
        num_updates=4,
        num_envs=1024,
    )
    wandb: WandbConfig = WandbConfig(
        enabled=True,
        project="modrax",
        group="lstm-vs-gtrxl-craftax",
    )
    eval_interval: int = 250
    eval_max_steps: int = 5000
    save_gif_wandb: bool = True
    gif_max_steps: int = 1000
    seed: int = 0


class CraftaxNetwork(PPONetwork):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        num_envs: int,
        cfg: CraftaxNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.is_recurrent = True
        self.architecture = cfg.architecture
        hidden_dim = cfg.lstm.hidden_dim if cfg.architecture == "lstm" else cfg.gtrxl.hidden_dim
        self.encoder = nnx.Linear(math.prod(obs_shape), hidden_dim, rngs=rngs)

        if cfg.architecture == "lstm":
            self.memory = NnxRNN(
                hidden_dim,
                hidden_dim,
                rngs,
                cell_type="lstm",
                num_layers=cfg.lstm.num_layers,
            )
        else:
            self.memory = GatedTransformerXL(
                input_dim=hidden_dim,
                num_heads=cfg.gtrxl.num_heads,
                num_layers=cfg.gtrxl.num_layers,
                rollout_memory_len=cfg.gtrxl.memory_len,
                segment_len=cfg.gtrxl.segment_len,
                gating=True,
                gating_bias=2.0,
                rngs=rngs,
            )
        self.memory.initialize_carry(num_envs)

        self.policy_ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.value_ln = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.policy_head = nnx.Linear(hidden_dim, num_actions, rngs=rngs)
        self.value_head = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def _apply_heads(self, x: Float[Array, "... D"]):
        policy_logits = self.policy_head(self.policy_ln(x))
        value = self.value_head(self.value_ln(x))
        return policy_logits, value

    def bootstrap_value(self, env_state: StateWithMetrics):
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self.encoder(obs)
        _, out = self.memory._eval_forward(encoded)
        if self.architecture == "gtrxl":
            out = out[:, 0]
        return self.value_head(self.value_ln(out))

    def train_forward(
        self, obs: Float[Array, "B T ..."], dones: Float[Array, "B T"], init_carry, saved_carry
    ):
        encoded = self.encoder(obs.reshape(*obs.shape[:2], -1))
        if self.architecture == "lstm":
            encoded = self.memory.train_forward(encoded, dones, init_carry)
        else:
            encoded = self.memory.train_forward(encoded, init_carry, saved_carry)

        policy_logits, value = self._apply_heads(encoded)
        return PPONetworkOutput(policy_logits, value, None)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self.encoder(obs)
        carry, out = self.memory(encoded)

        policy_logits, value = self._apply_heads(out)
        action = softmax_policy(policy_logits, env_state.action_mask, key)
        return action, PPONetworkOutput(policy_logits, value, carry)

    def reset(self):
        self.memory.reset()

    def reset_episodes(self, done: Array):
        self.memory.reset_episodes(done)

    def get_carry(self):
        return self.memory.carry.get_value()


@add_cli
def main(cfg: CraftaxRecurrentPPOConfig):
    key = jax.random.key(cfg.seed)
    network_key, alg_key, train_key = jax.random.split(key, 3)

    env = CraftaxEnv(cfg.env_cfg)
    network = CraftaxNetwork(
        env.obs_shape,
        env.action_size,
        cfg.alg_cfg.num_envs,
        cfg.network_cfg,
        nnx.Rngs(network_key),
    )
    alg = PPOAlg(env, network, cfg.alg_cfg, key=alg_key)
    train(alg, cfg, key=train_key)


if __name__ == "__main__":
    main()
