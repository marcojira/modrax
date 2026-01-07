"""Train MRQ on MinAtar."""

from modrax.alg.mrq import MRQAlg, MRQConfig, MRQNetwork, MRQNetworkConfig
from modrax.env import PGXEnvConfig
from modrax.network import MLP, BlockNetwork, BlockNetworkConfig, Linear, LinearConfig, MLPConfig
from modrax.optimizer import OptimizerConfig
from modrax.policy import softmax_policy
from modrax.rollout import RolloutConfig, rollout
from modrax.training import TrainConfig, train
from modrax.types import NUM_ACTIONS, OBS_SHAPE


def main():
    # Encoder configuration
    encoder_config = BlockNetworkConfig(
        encoders={"obs": (MLP, MLPConfig(hidden_dims=(128,)), OBS_SHAPE)},
        encoder_dim=64,
        trunk=(Linear, LinearConfig()),
        trunk_dim=32,
        heads={"state_encoding": (Linear, LinearConfig(), 64)},
    )

    # MRQ network configuration
    network_config = MRQNetworkConfig(
        encoder_cls=BlockNetwork,
        encoder_config=encoder_config,
        action_dim=32,
        state_action_encoder_config=MLPConfig(hidden_dims=(128, 128)),
        state_action_dim=64,
        value_config=MLPConfig(hidden_dims=(128, 128)),
        policy_config=MLPConfig(hidden_dims=(64, 64)),
    )

    mrq_config = MRQConfig(
        rollout_fn=rollout,
        rollout_config=RolloutConfig(num_steps=128),
        policy_fn=softmax_policy,
        buffer_size=100_000,
        minibatch_size=512,
        encoder_horizon=4,
        reward_horizon=4,
    )

    train_config = TrainConfig(
        # Environment suite/environment
        env_config=PGXEnvConfig(env_name="minatar-asterix", optimistic_reset=False),
        # Network class and config
        network_cls=MRQNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=3e-4, gradient_clip=0.5),
        # Algorithm
        alg=MRQAlg,
        alg_config=mrq_config,
        # Training parameters
        seed=0,
        num_envs=4096,
        total_steps=250_000_000,
        num_epochs=1,
        jit=True,
        # Save location
        save_path="out/examples/mrq-minatar-asterix",
        # display_network=True,
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
