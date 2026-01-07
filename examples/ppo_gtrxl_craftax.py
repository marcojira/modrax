"""Train PPO with GTrXL on Craftax."""

from modrax.alg.ppo import PPOAlg, PPOConfig
from modrax.env import CraftaxEnvConfig
from modrax.network import (
    MLP,
    GatedTransformerXL,
    GTrXLConfig,
    Linear,
    LinearConfig,
    MLPConfig,
    RecurrentNetwork,
    RecurrentNetworkConfig,
)
from modrax.optimizer import OptimizerConfig
from modrax.training import TrainConfig, train
from modrax.types import NUM_ACTIONS, OBS_SHAPE


def main():
    env_config = CraftaxEnvConfig(env_name="Craftax-Symbolic-v1", optimistic_reset=True)

    network_config = RecurrentNetworkConfig(
        encoders={"obs": (Linear, LinearConfig(), OBS_SHAPE)},
        encoder_dim=256,
        recurrent=(
            GatedTransformerXL,
            GTrXLConfig(
                num_heads=8,
                num_layers=2,
                gating_bias=2.0,
                segment_len=64,
                rollout_memory_len=128,
            ),
        ),
        recurrent_dim=256,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(256, 256)), NUM_ACTIONS),
            "value": (MLP, MLPConfig(hidden_dims=(256, 256)), 1),
        },
    )

    train_config = TrainConfig(
        seed=0,
        env_config=env_config,
        network_cls=RecurrentNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=2e-4, gradient_clip=0.5),
        alg_cls=PPOAlg,
        alg_config=PPOConfig(
            num_gen_steps=128,
            minibatch_size=128,
            gamma=0.999,
            gae_lambda=0.8,
            entropy_coeff=0.002,
            num_epochs=4,
        ),
        num_envs=1024,
        total_steps=1_000_000_000,
        jit=True,
        save_path="out/examples/gtrxl-craftax",
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
