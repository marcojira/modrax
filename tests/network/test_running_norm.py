import jax.numpy as jnp

from modrax.network.running_norm import RunningNorm


def test_running_norm_centers_a_batch():
    norm = RunningNorm(input_shape=2)
    batch = jnp.array([[1.0, 2.0], [3.0, 4.0]])

    norm.update(batch)
    normalized = norm(batch)

    assert norm.count[...] == 2
    assert jnp.allclose(normalized.mean(axis=0), 0)
    assert jnp.allclose(normalized.var(axis=0), 1)
