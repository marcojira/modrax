import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key


def softmax_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
) -> Float[Array, " B"]:
    masked_logits = jnp.where(action_mask, logits, -jnp.inf)
    action = jax.random.categorical(key, masked_logits)

    return action


def argmax_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
) -> Float[Array, " B"]:
    masked_logits = jnp.where(action_mask, logits, -jnp.inf)

    return masked_logits.argmax(axis=-1)


def uniform_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
) -> Float[Array, " B"]:
    """Sample uniformly from valid actions."""
    uniform_logits = jnp.where(action_mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, uniform_logits)
