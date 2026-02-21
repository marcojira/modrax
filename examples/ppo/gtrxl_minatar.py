"""Train PPO with GTrXL on MinAtar (faster alternative to Craftax for testing)."""

import math

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput
from modrax.env import Env, PGXConfig
from modrax.env.base import StateWithMetrics
from modrax.network.base import NetworkConfig
from modrax.network.gtrxl import GatedTransformerXL, GTrXLRecurrentState
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, train
from modrax.types import Shape


class MinAtarGTrXLNetworkConfig(NetworkConfig):
    encoder_dim: int = 32
    policy_hidden_dims: tuple[int, ...] = (32, 32)
    value_hidden_dims: tuple[int, ...] = (32, 32)
    num_heads: int = 4
    num_layers: int = 2
    gating_bias: float = 0.1
    segment_len: int = 8
    rollout_memory_len: int = 16


class MinAtarGTrXLNetwork(PPONetwork):
    def __init__(
        self,
        obs_shape: Shape,
        num_actions: int,
        num_envs: int,
        cfg: MinAtarGTrXLNetworkConfig,
        rngs: nnx.Rngs,
    ):
        relu = jax.nn.relu
        self.encoder = nnx.Linear(math.prod(obs_shape), cfg.encoder_dim, rngs=rngs)
        self.gtrxl = GatedTransformerXL(
            input_dim=cfg.encoder_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            rollout_memory_len=cfg.rollout_memory_len,
            segment_len=cfg.segment_len,
            rngs=rngs,
            gating=True,
            gating_bias=cfg.gating_bias,
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

    def bootstrap_value(self, env_state: StateWithMetrics) -> Float[Array, "B 1"]:
        """Compute value for GAE bootstrap without advancing the carry."""
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self.encoder(obs)
        _, out = self.gtrxl._eval_forward(encoded)
        return self.value_head(out[:, 0, :])

    def reset(self, done: Float[Array, " B"]):
        self.gtrxl.reset(done)

    def get_carry(self):
        return self.gtrxl.carry.value


def main():
    env_config = PGXConfig(env_name="minatar-asterix", optimistic_reset=False)
    network_config = MinAtarGTrXLNetworkConfig()
    optimizer_config = OptimizerConfig(learning_rate=3e-4, gradient_clip=10)
    alg_config = PPOConfig(
        num_gen_steps=128,
        minibatch_size=128,
        num_epochs=3,
        num_envs=4096,
    )
    train_config = TrainConfig(
        seed=0,
        env_config=env_config,
        network_config=network_config,
        optimizer_config=optimizer_config,
        alg_config=alg_config,
        total_steps=250_000_000,
        jit=True,
        save_path="out/examples/gtrxl-minatar",
    )

    env = Env(env_config)
    network = MinAtarGTrXLNetwork(
        env.obs_shape, env.action_size, alg_config.num_envs, network_config, rngs=nnx.Rngs(0)
    )
    optimizer = Optimizer(optimizer_config, network)
    alg = PPOAlg(
        env, network, optimizer, alg_config, jax.random.key(train_config.seed), jit=train_config.jit
    )

    trained_network = train(env, network, optimizer, alg, train_config)
    return trained_network


if __name__ == "__main__":
    main()
