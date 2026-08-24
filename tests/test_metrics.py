import jax.numpy as jnp

from modrax.metrics import compute_training_metrics, finite_mean
from tests.helpers import make_trajectory


def test_compute_training_metrics_aggregates_completed_episodes():
    trajectory = make_trajectory(
        rewards=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        episode_returns=jnp.array([[0.0, 3.0], [4.0, 0.0]]),
        episode_lengths=jnp.array([[0, 2], [1, 0]]),
        dones=jnp.array([[False, True], [True, False]]),
    )

    metrics = compute_training_metrics(trajectory)

    assert metrics["Rew."] == 5
    assert metrics["Ep.Ret."] == 3.5
    assert metrics["Ep.Len."] == 1.5


def test_finite_mean_ignores_negative_infinity():
    assert finite_mean(jnp.array([1.0, -jnp.inf, 3.0])) == 2
