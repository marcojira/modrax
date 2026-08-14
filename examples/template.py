"""Template for implementing a Modrax algorithm."""

import math
from dataclasses import dataclass
from typing import Any

import jax
from flax import nnx, struct
from jaxtyping import Array, Key

from modrax.alg.base import Alg
from modrax.env.base import Env, StateWithMetrics
from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
from modrax.network import Network
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Config as BaseConfig
from modrax.types import Shape
from modrax.utils import add_cli


@dataclass(frozen=True)
class NetworkConfig(BaseConfig):
    hidden_dims: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class AlgConfig(BaseConfig):
    total_steps: int = 100_000
    num_envs: int = 16
    num_timesteps: int = 128


@dataclass(frozen=True)
class Config(TrainConfig):
    env_cfg: GymnaxConfig = GymnaxConfig(env_name="CartPole-v1")
    network_cfg: NetworkConfig = NetworkConfig()
    optimizer_cfg: OptimizerConfig = OptimizerConfig()
    alg_cfg: AlgConfig = AlgConfig()
    wandb: WandbConfig = WandbConfig(enabled=False, project="modrax")
    seed: int = 0


class MyNetwork(Network):
    def __init__(self, obs_shape: Shape, num_actions: int, cfg: NetworkConfig, rngs: nnx.Rngs):
        self.model = MLP(
            math.prod(obs_shape),
            cfg.hidden_dims,
            num_actions,
            activation_fn=jax.nn.relu,
            rngs=rngs,
        )

    def __call__(self, obs: Array) -> Array:
        return self.model(obs.reshape(obs.shape[0], -1))

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        logits = self(env_state.obs)
        action = softmax_policy(logits, env_state.action_mask, key)
        return action, logits


@struct.dataclass
class MyAlgState:
    agent_state: Any
    env_state: StateWithMetrics
    step: int


def loss_fn(network: MyNetwork, minibatch, config: AlgConfig):
    """Compute the algorithm-specific loss for one minibatch."""
    raise NotImplementedError


class MyAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: MyNetwork,
        optimizer: Optimizer,
        cfg: AlgConfig,
        key: Key[Array, ""],
    ):
        super().__init__(env, network, optimizer, cfg)
        self.total_steps = cfg.total_steps
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_timesteps

        env_state = env.reset(jax.random.split(key, cfg.num_envs))
        self.state = MyAlgState(nnx.split((network, optimizer)), env_state, 0)

    def __call__(self, key: Key[Array, ""]) -> dict[str, float]:
        """Run one training epoch and update ``self.state``."""
        raise NotImplementedError


@add_cli
def main(cfg: Config):
    key = jax.random.key(cfg.seed)
    network_key, alg_key, train_key = jax.random.split(key, 3)

    env = GymnaxEnv(cfg.env_cfg)
    network = MyNetwork(env.obs_shape, env.action_size, cfg.network_cfg, nnx.Rngs(network_key))
    optimizer = Optimizer(cfg.optimizer_cfg, network)
    alg = MyAlg(env, network, optimizer, cfg.alg_cfg, key=alg_key)

    train(alg, cfg, key=train_key)


if __name__ == "__main__":
    main()
