"""Train Gumbel AlphaZero on Tic Tac Toe."""

from modrax.env import PGXEnvConfig
from modrax.eval import VsUniformConfig, eval_vs_uniform
from modrax.network import AZNet, AZNetConfig
from modrax.optimizer import OptimizerConfig
from modrax.policy import softmax_policy
from modrax.rollout import RolloutConfig, rollout
from modrax.training import TrainConfig, train
from modrax.update.gumbel_az import GumbelAZConfig, gumbel_az_update


def main():
    env_config = PGXEnvConfig(env_name="connect_four")
    network_config = AZNetConfig(num_channels=64, num_blocks=6, resnet_v2=True)

    train_config = TrainConfig(
        env_config=env_config,
        network_cls=AZNet,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=1e-3, gradient_clip=10),
        rollout_fn=rollout,
        rollout_config=RolloutConfig(num_steps=128),
        update_fn=gumbel_az_update,
        update_config=GumbelAZConfig(minibatch_size=4096),
        policy_fn=softmax_policy,
        eval_fn=eval_vs_uniform,  # type: ignore
        eval_config=VsUniformConfig(env_config=env_config, num_games=1024),
        eval_interval=50,
        log_interval=50,
        seed=0,
        num_envs=1024,
        total_steps=250_000_000,
        num_epochs=3,
        jit=True,
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
