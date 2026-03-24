import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx, struct

from modrax.alg.base import Alg
from modrax.env.base import Env, StateWithMetrics
from modrax.env.gymnax import GymnaxConfig, GymnaxEnv
from modrax.network import Network
from modrax.network.mlp import MLP
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import epsilon_greedy_policy
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
class NetworkConfig:
    hidden_dims: tuple[int, ...] = (128, 128)


@dataclass(frozen=True)
class AlgConfig:
    eps: float = 0.05

    num_envs: int = 4096
    num_gen_steps: int = 128
    minibatch_size: int = 4096

    gamma: float = 0.99
    total_steps: int = 100_000_000


@dataclass(frozen=True)
class Config(TrainConfig):
    env_cfg: GymnaxConfig = GymnaxConfig(env_name="Asterix-MinAtar")
    network_cfg: NetworkConfig = NetworkConfig()
    optimizer_cfg: OptimizerConfig = OptimizerConfig(
        optimizer_type="adam", learning_rate=3e-4, gradient_clip=2.0
    )
    wandb: WandbConfig = WandbConfig(enabled=True, project="modrax")
    alg_cfg: AlgConfig = AlgConfig()
    eval_interval: int = 25
    seed: int = 0
    save_gif_wandb: bool = True
    num_gif_trajectories: int = 2


""" NETWORK """


class DQNNetwork(Network):
    def __init__(self, obs_shape, num_actions: int, cfg: NetworkConfig, eps: float, rngs: nnx.Rngs):
        input_dim = math.prod(obs_shape)
        self.q_network = MLP(input_dim, cfg.hidden_dims, num_actions, jax.nn.relu, rngs)
        self.eps = eps

    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.q_network(x)

    def policy(self, env_state, key):
        q_values = self.__call__(env_state.obs)
        action = epsilon_greedy_policy(q_values, env_state.action_mask, key, self.eps)
        return action, q_values


""" ALGORITHM """


@struct.dataclass
class DQNState:
    agent_state: Any
    env_state: StateWithMetrics
    step: int


def dqn_loss(network: DQNNetwork, minibatch, config: AlgConfig):
    q_values = network(minibatch.obs)
    next_q_values = network(minibatch.next_obs)
    next_q_values = jax.lax.stop_gradient(next_q_values)

    q_target = minibatch.rewards + config.gamma * (1 - minibatch.dones) * next_q_values.max(axis=-1)
    chosen_q_values = jnp.take_along_axis(q_values, minibatch.actions[..., None], axis=-1)
    chosen_q_values = chosen_q_values.squeeze(axis=-1)

    loss = 0.5 * jnp.square(chosen_q_values - q_target).mean()
    return loss, {"loss": loss}


class DQNAlg(Alg):
    def __init__(
        self, env: Env, network: Network, optimizer: Optimizer, cfg: AlgConfig, key: jax.Array
    ):
        super().__init__(env, network, optimizer, cfg, key)
        self.total_steps = cfg.total_steps
        self.env_steps_per_epoch = cfg.num_envs * cfg.num_gen_steps

        env_state = self.env.reset(jax.random.split(key, self.cfg.num_envs))
        self.state = DQNState(nnx.split((self.network, self.optimizer)), env_state, 0)

        self.main_loop = nnx.jit(self._step)

    def _step(self, state: DQNState, key):
        key, rollout_key, update_key = jax.random.split(key, 3)
        network, optimizer = nnx.merge(*state.agent_state)

        # Gen data
        new_env_state, transitions = transitions_rollout(
            network, self.env.step, state.env_state, self.cfg.num_gen_steps, rollout_key
        )

        # Train
        minibatch_key, update_key = jax.random.split(update_key)
        minibatches = make_transition_minibatches(
            transitions, minibatch_key, self.cfg.minibatch_size
        )
        loss, infos = update_network_minibatches(
            network, optimizer, minibatches, dqn_loss, self.cfg
        )

        metrics = compute_training_metrics(transitions)
        metrics.update(jax.tree.map(lambda x: x.mean(), infos))

        return (
            metrics,
            DQNState(nnx.split((network, optimizer)), new_env_state, state.step + 1),
        )

    def __call__(self, key: jax.Array) -> dict[str, float]:
        metrics, new_state = self.main_loop(self.state, key)
        self.state = new_state

        return metrics

    def eval(self, key: jax.Array):
        network, _ = nnx.merge(*self.state.agent_state)
        network = nnx.clone(network)
        network.eps = 0.0
        return eval_rollout(self.env, network, self.cfg.num_envs, key, max_steps=2000)

@add_cli
def main(cfg: Config):
    key = jax.random.key(cfg.seed)

    env = GymnaxEnv(cfg.env_cfg)
    network = DQNNetwork(
        env.obs_shape, env.action_size, cfg.network_cfg, cfg.alg_cfg.eps, nnx.Rngs(cfg.seed)
    )
    optimizer = Optimizer(cfg.optimizer_cfg, network)
    alg = DQNAlg(env, network, optimizer, cfg.alg_cfg, key)

    network = train(env, network, optimizer, alg, cfg)
    return network


if __name__ == "__main__":
    main()
