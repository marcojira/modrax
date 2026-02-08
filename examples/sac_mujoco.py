"""Train PPO on MinAtar (250M steps, ~100s on an L40s)"""

from modrax.alg.sac import SACAlg, SACConfig, SACNetworkConfig
from modrax.env import MuJoCoEnvConfig
from modrax.network import MLP, BlockNetwork, BlockNetworkConfig, Linear, LinearConfig, MLPConfig
from modrax.optimizer import OptimizerConfig
from modrax.training import TrainConfig, train
from modrax.types import NUM_ACTIONS, OBS_SHAPE


def main():
    train_config = TrainConfig(
        # Environment suite/environment
        env_config=MuJoCoEnvConfig(env_name="HopperHop"),
        # Network class and config
        # Specify algorithm class and config
        alg_cls=SACAlg,
        alg_config=SACConfig(num_gen_steps=1),
        # Training parameters
        seed=0,
        num_envs=128,
        total_steps=250_000_000,
        jit=True,
        # Save location
        # save_path="out/examples/sac_mujoco",
        save_path=None,
        eval_interval=5000,
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
