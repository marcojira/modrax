"""Train Gumbel AlphaZero on Connect Four."""

from modrax.alg.gumbel_az import GumbelAZAlg, GumbelAZConfig
from modrax.env import PGXEnvConfig
from modrax.eval import VsUniformConfig, eval_vs_uniform
from modrax.network import AZNet, AZNetConfig
from modrax.optimizer import OptimizerConfig
from modrax.training import TrainConfig, train


def main():
    env_config = PGXEnvConfig(env_name="connect_four")
    network_config = AZNetConfig(num_channels=64, num_blocks=6, resnet_v2=True)

    train_config = TrainConfig(
        env_config=env_config,
        network_cls=AZNet,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=1e-3, gradient_clip=10),
        alg_cls=GumbelAZAlg,
        alg_config=GumbelAZConfig(
            num_gen_steps=128,
            minibatch_size=128,
            num_epochs=3,
        ),
        eval_fn=eval_vs_uniform,  # type: ignore
        eval_config=VsUniformConfig(env_config=env_config, num_games=1024),
        eval_interval=25,
        seed=0,
        num_envs=1024,
        total_steps=250_000_000,
        jit=True,
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
