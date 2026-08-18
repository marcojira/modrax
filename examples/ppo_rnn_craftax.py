"""Train PPO with RNN on Craftax."""

import math
from dataclasses import dataclass

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.base import OptimizerConfig
from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput
from modrax.cli import add_cli
from modrax.env.base import StateWithMetrics
from modrax.env.craftax import CraftaxConfig, CraftaxEnv
from modrax.network import NetworkConfig
from modrax.network.rnn import NnxRNN
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train


@dataclass(frozen=True)
class CraftaxRNNNetworkConfig(NetworkConfig):
    encoder_dim: int = 512
    rnn_hidden_dim: int = 512
    cell_type: str = "lstm"
    num_rnn_layers: int = 1


@dataclass(frozen=True)
class CraftaxRNNPPOConfig(TrainConfig):
    env_cfg: CraftaxConfig = CraftaxConfig(
        env_name="Craftax-Symbolic-v1",
        optimistic_reset=True,
    )
    network_cfg: CraftaxRNNNetworkConfig = CraftaxRNNNetworkConfig()
    alg_cfg: PPOConfig = PPOConfig(
        optimizer_cfg=OptimizerConfig(learning_rate=2e-4, gradient_clip=1.0, lr_decay=True),
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
    eval_max_steps: int = 2500
    seed: int = 0
    save_gif_wandb: bool = True
    num_gif_trajectories: int = 2


class CraftaxRNNNetwork(PPONetwork):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        num_envs: int,
        cfg: CraftaxRNNNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.is_recurrent = True

        self.encoder = nnx.Linear(math.prod(obs_shape), cfg.encoder_dim, rngs=rngs)
        self.rnn = NnxRNN(
            input_dim=cfg.encoder_dim,
            output_dim=cfg.rnn_hidden_dim,
            cell_type=cfg.cell_type,
            num_layers=cfg.num_rnn_layers,
            rngs=rngs,
        )
        self.rnn.initialize_carry(num_envs)

        self.policy_ln = nnx.LayerNorm(cfg.rnn_hidden_dim, rngs=rngs)
        self.value_ln = nnx.LayerNorm(cfg.rnn_hidden_dim, rngs=rngs)
        self.policy_head = nnx.Linear(cfg.rnn_hidden_dim, num_actions, rngs=rngs)
        self.value_head = nnx.Linear(cfg.rnn_hidden_dim, 1, rngs=rngs)

    def _apply_heads(self, x: Float[Array, "... D"]):
        policy_logits = self.policy_head(self.policy_ln(x))
        value = self.value_head(self.value_ln(x))
        return policy_logits, value

    def train_forward(
        self, obs: Float[Array, "B T ..."], dones: Float[Array, "B T"], init_carry, saved_carry
    ):
        encoded = self.encoder(obs.reshape(*obs.shape[:2], -1))
        encoded = self.rnn.train_forward(encoded, dones, init_carry)

        policy_logits, value = self._apply_heads(encoded)
        return PPONetworkOutput(policy_logits, value, None)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self.encoder(obs)
        carry, out = self.rnn(encoded)

        policy_logits, value = self._apply_heads(out)
        action = softmax_policy(policy_logits, env_state.action_mask, key)
        return action, PPONetworkOutput(policy_logits, value, carry)

    def reset(self):
        self.rnn.reset()

    def reset_episodes(self, done: Array):
        self.rnn.reset_episodes(done)

    def get_carry(self):
        return self.rnn.carry.value


@add_cli
def main(cfg: CraftaxRNNPPOConfig):
    key = jax.random.key(cfg.seed)
    network_key, alg_key, train_key = jax.random.split(key, 3)

    # Init objects
    env = CraftaxEnv(cfg.env_cfg)
    network = CraftaxRNNNetwork(
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
