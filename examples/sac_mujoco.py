import os

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

from modrax.alg.sac import SACAlg, SACConfig, SACNetworkConfig
from modrax.env import MuJoCoEnvConfig
from modrax.training import TrainConfig, train


def main():
    train_config = TrainConfig(
        # Environment suite/environment
        env_config=MuJoCoEnvConfig(env_name="HumanoidWalk"),
        # Network class and config
        alg_cls=SACAlg,
        alg_config=SACConfig(
            num_gen_steps=1000, network_config=SACNetworkConfig(running_norm=True)
        ),
        # Training parameters
        seed=0,
        num_envs=128,
        total_steps=100_000_000,
        jit=True,
        # Save location
        save_path=None,
        eval_interval=5000,
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
