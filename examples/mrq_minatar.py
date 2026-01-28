"""Train MRQ on MinAtar."""

import jax

from modrax.alg.mrq import MRQAlg, MRQConfig, MRQNetwork, MRQNetworkConfig, mrq_policy
from modrax.env import PGXEnvConfig
from modrax.eval.rollout_return import RolloutReturnConfig, rollout_return
from modrax.optimizer import OptimizerConfig
from modrax.rollout.rollout import rollout
from modrax.training import TrainConfig, train


def main():
    network_config = MRQNetworkConfig(
        pixel_obs=True,
        zs_dim=512,
        za_dim=256,
        zsa_dim=512,
        enc_hdim=512,
        value_hdim=512,
        policy_hdim=512,
        enc_activation=jax.nn.elu,
        value_activation=jax.nn.elu,
        policy_activation=jax.nn.relu,
        num_bins=65,
    )

    mrq_config = MRQConfig(
        buffer_size=1_000_000,
        minibatch_size=256,
        encoder_horizon=5,
        reward_horizon=3,
        num_gen_steps=256,
    )

    train_config = TrainConfig(
        # Environment suite/environment
        env_config=PGXEnvConfig(env_name="minatar-asterix", optimistic_reset=False),
        # Network class and config
        network_cls=MRQNetwork,
        network_config=network_config,
        optimizer_config=OptimizerConfig(learning_rate=3e-4, gradient_clip=100),
        # Algorithm
        alg_cls=MRQAlg,
        alg_config=mrq_config,
        # Eval
        eval_interval=25,
        eval_fn=rollout_return,
        eval_config=RolloutReturnConfig(rollout_fn=rollout, policy_fn=mrq_policy),
        # Training parameters
        seed=0,
        num_envs=1,
        total_steps=250_000_000,
        jit=True,
        # Save location
        save_path="out/examples/mrq-minatar-asterix",
        # display_network=True,
    )

    trained_network = train(train_config)
    return trained_network


if __name__ == "__main__":
    main()
