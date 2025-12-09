"""Train PPO on MinAtar."""

from modrax.env import PGXEnvConfig
from modrax.network import MLP, BlockNetwork, BlockNetworkConfig, Linear, LinearConfig, MLPConfig
from modrax.optimizer import OptimizerConfig
from modrax.policy import softmax_policy
from modrax.rollout import RolloutConfig, rollout
from modrax.training import TrainConfig, train
from modrax.update import PPOConfig, ppo_update


def main():
    env_config = PGXEnvConfig(env_name="minatar-asterix")

    network_config = BlockNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(128,)))},
        encoder_dim=64,
        trunk=(Linear, LinearConfig()),
        trunk_dim=32,
        heads={
            "policy": (MLP, MLPConfig(hidden_dims=(64, 64))),
            "value": (MLP, MLPConfig(hidden_dims=(64, 64))),
        },
    )

    train_config = TrainConfig(
        seed=0,
        env_config=env_config,
        network_cls=BlockNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=3e-4, gradient_clip=0.5),
        rollout_fn=rollout,
        rollout_config=RolloutConfig(num_steps=128),
        update_fn=ppo_update,
        update_config=PPOConfig(minibatch_size=128),
        policy_fn=softmax_policy,
        num_envs=4096,
        total_steps=250_000_000,
        num_epochs=3,
        jit=True,
        save_path="out/examples/minatar-asterix",
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
