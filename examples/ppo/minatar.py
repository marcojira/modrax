"""Train PPO on MinAtar (250M steps, ~100s on an L40s)"""

import math

import jax
from flax import nnx
from jaxtyping import Array, Float, Key

from modrax.alg.ppo import PPOAlg, PPOConfig, PPONetwork, PPONetworkOutput
from modrax.env import Env, PGXConfig
from modrax.env.base import StateWithMetrics
from modrax.network.base import NetworkConfig
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, train
from modrax.types import Shape


class MinAtarNetworkConfig(NetworkConfig):
    encoder_hidden_dims: tuple[int, ...] = (128,)
    encoder_dim: int = 64
    policy_hidden_dims: tuple[int, ...] = (64, 64)
    value_hidden_dims: tuple[int, ...] = (64, 64)


class MinAtarNetwork(PPONetwork):
    def __init__(
        self, obs_shape: Shape, num_actions: int, cfg: MinAtarNetworkConfig, rngs: nnx.Rngs
    ):
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

    def bootstrap_value(self, env_state: StateWithMetrics) -> Float[Array, "B 1"]:
        _, value = self._forward(self._flatten_obs(env_state.obs))
        return value


def main():
    # Configs
    env_config = PGXConfig(env_name="minatar-asterix", optimistic_reset=False)
    network_config = MinAtarNetworkConfig()
    optimizer_config = OptimizerConfig(learning_rate=3e-4, gradient_clip=0.5)
    alg_config = PPOConfig(num_gen_steps=128, minibatch_size=128, num_epochs=3, num_envs=4096)
    train_config = TrainConfig(
        env_config=env_config,
        network_config=network_config,
        optimizer_config=optimizer_config,
        alg_config=alg_config,
        # Training parameters
        seed=0,
        total_steps=250_000_000,
        jit=True,
        # Save location
        save_path="out/examples/minatar-asterix",
    )

    # Init
    env = Env(env_config)
    network = MinAtarNetwork(
        env.obs_shape, env.action_size, network_config, rngs=nnx.Rngs(train_config.seed)
    )
    optimizer = Optimizer(optimizer_config, network)
    alg = PPOAlg(
        env, network, optimizer, alg_config, jax.random.key(train_config.seed), jit=train_config.jit
    )

    # Train
    trained_network = train(env, network, optimizer, alg, train_config)
    return trained_network


if __name__ == "__main__":
    main()
