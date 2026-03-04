"""Train MRQ on MinAtar."""

import jax
from flax import nnx

from modrax.alg.mrq import MRQAlg, MRQConfig, MRQNetwork, MRQNetworkConfig
from modrax.env import Env, PGXConfig
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.training import TrainConfig, train


def main():
    env_config = PGXConfig(env_name="minatar-asterix", optimistic_reset=False)
    network_config = MRQNetworkConfig(
        pixel_obs=True,
    )
    optimizer_config = OptimizerConfig(learning_rate=3e-4, gradient_clip=100)
    alg_config = MRQConfig(
        buffer_size=1_000_000,
        minibatch_size=256,
        encoder_horizon=5,
        reward_horizon=3,
        num_gen_steps=256,
    )
    train_config = TrainConfig(
        seed=0,
        env_config=env_config,
        network_config=network_config,
        optimizer_config=optimizer_config,
        alg_config=alg_config,
        total_steps=10_000_000,
        save_path="out/examples/mrq-minatar-asterix",
    )

    env = Env(env_config)
    network = MRQNetwork(
        env.obs_shape, env.action_size, network_config, nnx.Rngs(train_config.seed)
    )
    optimizer = Optimizer(optimizer_config, network)
    alg = MRQAlg(
        env, network, optimizer, alg_config, jax.random.key(train_config.seed)
    )

    trained_network = train(env, network, optimizer, alg, train_config)
    return trained_network


if __name__ == "__main__":
    main()
