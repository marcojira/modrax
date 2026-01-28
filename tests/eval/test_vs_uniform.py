"""Test eval_vs_uniform function."""

from modrax.env import Env, PGXEnvConfig
from modrax.eval import VsUniformConfig, eval_vs_uniform
from modrax.network.block_network import BlockNetwork
from modrax.network.recurrent_network import RecurrentNetwork


def test_eval_vs_uniform(rng_key, rngs, block_network_cfg):
    """Test that eval returns valid win/draw/loss rates that sum to 1."""
    env_config = PGXEnvConfig(env_name="tic_tac_toe")
    env = Env(env_config, jit=False)

    network = BlockNetwork(
        obs_shape=env.obs_shape,
        num_actions=env.num_actions,
        config=block_network_cfg,
        rngs=rngs,
    )

    config = VsUniformConfig(env_config=env_config, num_games=20)
    results = eval_vs_uniform(network, env, config, rng_key)  # type: ignore

    assert 0.0 <= results["win_rate"] <= 1.0
    assert 0.0 <= results["draw_rate"] <= 1.0
    assert 0.0 <= results["loss_rate"] <= 1.0
    assert abs(results["win_rate"] + results["draw_rate"] + results["loss_rate"] - 1.0) < 1e-6


def test_eval_vs_uniform_recurrent(rng_key, rngs, recurrent_network_cfg):
    """Test eval with recurrent networks."""
    env_config = PGXEnvConfig(env_name="tic_tac_toe")
    env = Env(env_config, jit=False)

    network = RecurrentNetwork(
        obs_shape=env.obs_shape,
        num_actions=env.num_actions,
        config=recurrent_network_cfg,
        rngs=rngs,
    )

    eval_config = VsUniformConfig(env_config=env_config, num_games=20)
    results = eval_vs_uniform(network, env, eval_config, rng_key)  # type: ignore

    assert 0.0 <= results["win_rate"] <= 1.0
    assert 0.0 <= results["draw_rate"] <= 1.0
    assert 0.0 <= results["loss_rate"] <= 1.0
    assert abs(results["win_rate"] + results["draw_rate"] + results["loss_rate"] - 1.0) < 1e-6
