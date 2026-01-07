"""Train PPO on MinAtar (250M steps, ~100s on an L40s)"""

from modrax.alg.ppo import PPOAlg, PPOConfig
from modrax.env import PGXEnvConfig
from modrax.network import MLP, BlockNetwork, BlockNetworkConfig, Linear, LinearConfig, MLPConfig
from modrax.optimizer import OptimizerConfig
from modrax.training import TrainConfig, train
from modrax.types import NUM_ACTIONS, OBS_SHAPE


def main():
    # Networks are built from blocks: BlockNetwork goes from encoders -> trunk -> heads
    network_config = BlockNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(128,)), OBS_SHAPE)},
        encoder_dim=64,
        trunk=(Linear, LinearConfig()),
        trunk_dim=32,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(64, 64)), NUM_ACTIONS),
            "value": (MLP, MLPConfig(hidden_dims=(64, 64)), 1),
        },
    )

    train_config = TrainConfig(
        # Environment suite/environment
        env_config=PGXEnvConfig(env_name="minatar-asterix", optimistic_reset=False),
        # Network class and config
        network_cls=BlockNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=3e-4, gradient_clip=0.5),
        # Specify algorithm class and config
        alg_cls=PPOAlg,
        alg_config=PPOConfig(
            num_gen_steps=128,
            minibatch_size=128,
        ),
        # Training parameters
        seed=0,
        num_envs=4096,
        total_steps=250_000_000,
        num_epochs=3,
        jit=True,
        # Save location
        save_path="out/examples/minatar-asterix",
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
