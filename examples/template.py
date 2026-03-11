from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx, struct

from modrax.alg.base import Alg
from modrax.env.base import Env, StateWithMetrics
from modrax.network import Network
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.rollout.eval_rollout import eval_rollout
from modrax.rollout.transitions_rollout import transitions_rollout
from modrax.training import TrainConfig, WandbConfig, train
from modrax.utils import (
    add_cli,
    compute_training_metrics,
    make_transition_minibatches,
    update_network_minibatches,
)

""" CONFIG """


@dataclass(frozen=True)
class NetworkConfig: ...


@dataclass(frozen=True)
class AlgConfig: ...


@dataclass(frozen=True)
class Config(TrainConfig):
    env_cfg: ... = ...
    network_cfg: NetworkConfig = NetworkConfig()
    optimizer_cfg: OptimizerConfig = ...
    alg_cfg: AlgConfig = AlgConfig()
    wandb_cfg: WandbConfig = ...

    seed: int = 0


""" NETWORK """


class MyNetwork(Network):
    def __init__(self, obs_shape, num_actions: int, cfg: NetworkConfig, rngs: nnx.Rngs): ...

    def __call__(self, x): ...

    def policy(self, env_state, key): ...


""" ALGORITHM """


@struct.dataclass
class MyAlgState: ...


def loss_fn(network: MyNetwork, minibatch, config: AlgConfig): ...


class MyAlg(Alg):
    def __init__(
        self, env: Env, network: Network, optimizer: Optimizer, cfg: AlgConfig, key: jax.Array
    ):
        super().__init__(env, network, optimizer, cfg, key)
        ...

    def _step(self, state: MyAlgState, key): ...

    def __call__(self, key: jax.Array) -> dict[str, float]: ...

    def eval(self, key: jax.Array): ...

@add_cli
def main(cfg: Config):
    key = jax.random.key(cfg.seed)

    env = ...
    network = ...
    optimizer = ...
    alg = ...

    # TODO: Uncomment once done!
    # network = train(env, network, optimizer, alg, cfg)
    return network


if __name__ == "__main__":
    main()
