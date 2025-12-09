"""Train PPO with GTrXL on Craftax."""

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
from modrax.policy import softmax_policy
from modrax.rollout import RolloutConfig, recurrent_rollout
from modrax.training import TrainConfig, train
from modrax.update import PPOConfig, ppo_update


def main():
    env_config = CraftaxEnvConfig(env_name="Craftax-Symbolic-v1")

    network_config = RecurrentNetworkConfig(
        encoders={"obs": (Linear, LinearConfig())},
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
            "policy": (MLP, MLPConfig(hidden_dims=(256, 256))),
            "value": (MLP, MLPConfig(hidden_dims=(256, 256))),
        },
    )

    train_config = TrainConfig(
        seed=0,
        env_config=env_config,
        network_cls=RecurrentNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=2e-4, gradient_clip=0.5),
        rollout_fn=recurrent_rollout,
        rollout_config=RolloutConfig(num_steps=128),
        update_fn=ppo_update,
        update_config=PPOConfig(
            minibatch_size=128,
            gamma=0.999,
            gae_lambda=0.8,
            entropy_coeff=0.002,
        ),
        policy_fn=softmax_policy,
        num_envs=1024,
        total_steps=1_000_000_000,
        num_epochs=4,
        jit=True,
        save_path="out/examples/gtrxl-craftax",
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
