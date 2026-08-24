import jax

from modrax.eval import eval_rollout
from tests.helpers import ConstantPolicy, CountingEnv


def test_eval_rollout_reports_episode_metrics_and_trajectories():
    env = CountingEnv(episode_length=2, auto_reset=False)

    metrics, trajectories = eval_rollout(
        env,
        ConstantPolicy(),
        num_envs=2,
        key=jax.random.key(0),
        max_steps=3,
        num_trajectories=1,
    )

    assert metrics["eval_return"] == 2
    assert metrics["eval_length"] == 2
    assert trajectories.obs.shape == (1, 3, 1)
