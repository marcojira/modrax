"""Template for implementing a Modrax algorithm."""

import math
from dataclasses import dataclass
from typing import Any

import jax
from flax import nnx, struct
from jaxtyping import Array, Key

from modrax.alg.base import Alg, AlgConfig, OptimizerConfig, create_optimizer
from modrax.cli import add_cli
from modrax.env.base import Env, StateWithMetrics
from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
from modrax.network import Network, NetworkConfig
from modrax.network.mlp import MLP
from modrax.policy import softmax_policy
from modrax.training import TrainConfig, WandbConfig, train


@dataclass(frozen=True)
class MyNetworkConfig(NetworkConfig):
    hidden_dims: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class MyAlgConfig(AlgConfig):
    optimizer_cfg: OptimizerConfig = OptimizerConfig()
    total_steps: int = 100_000
    num_envs: int = 16
    num_timesteps: int = 128


@dataclass(frozen=True)
class Config(TrainConfig):
    env_cfg: GymnaxConfig = GymnaxConfig(env_name="CartPole-v1")
    network_cfg: MyNetworkConfig = MyNetworkConfig()
    alg_cfg: MyAlgConfig = MyAlgConfig()
    wandb: WandbConfig = WandbConfig(enabled=False, project="modrax")
    seed: int = 0


class MyNetwork(Network):
    def __init__(self, obs_shape: tuple[int, ...], num_actions: int, cfg: MyNetworkConfig, rngs: nnx.Rngs):
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


def loss_fn(network: MyNetwork, minibatch, config: MyAlgConfig):
    """Compute the algorithm-specific loss for one minibatch."""
    raise NotImplementedError


class MyAlg(Alg):
    def __init__(
        self,
        env: Env,
        network: MyNetwork,
        cfg: MyAlgConfig,
        key: Key[Array, ""],
    ):
        super().__init__(env, network, cfg)
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_timesteps

        optimizer = create_optimizer(network, cfg.optimizer_cfg)
        env_state = env.reset(jax.random.split(key, cfg.num_envs))
        self.state = MyAlgState(nnx.split((network, optimizer)), env_state, 0)
        self.jitted_step = nnx.jit(self._step_fn)

    def _step_fn(
        self, state: MyAlgState, key: Key[Array, ""]
    ) -> tuple[MyAlgState, dict[str, float]]:
        """Run one training epoch: collect data, update the network, return the new state."""
        rollout_key, update_key = jax.random.split(key)
        network, optimizer = nnx.merge(*state.agent_state)

        # 1. Collect data from ``state.env_state`` with ``rollout_key``, keeping the new env state.
        # 2. Update the network with ``update_key``, e.g. via ``update_network_minibatches``.
        # 3. Return ``MyAlgState(nnx.split((network, optimizer)), env_state, state.step + 1)``
        #    together with a metrics dict.
        raise NotImplementedError

    def step(self, key: Key[Array, ""]) -> dict[str, float]:
        self.state, metrics = self.jitted_step(self.state, key)
        return metrics


@add_cli
def main(cfg: Config):
    key = jax.random.key(cfg.seed)
    network_key, alg_key, train_key = jax.random.split(key, 3)

    env = GymnaxEnv(cfg.env_cfg)
    network = MyNetwork(env.obs_shape, env.action_size, cfg.network_cfg, nnx.Rngs(network_key))
    alg = MyAlg(env, network, cfg.alg_cfg, key=alg_key)

    train(alg, cfg, key=train_key)


if __name__ == "__main__":
    main()
