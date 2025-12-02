"""Shared pytest fixtures for tests."""

import jax
import pytest


@pytest.fixture
def rng_key():
    """Basic JAX random key."""
    return jax.random.key(0)
