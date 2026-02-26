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


def epsilon_greedy_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
    epsilon: float = 0.01,
) -> Float[Array, " B"]:
    random_key, explore_key = jax.random.split(key)
    greedy_action = argmax_policy(logits, action_mask, key)
    random_action = uniform_policy(logits, action_mask, random_key)
    explore = jax.random.uniform(explore_key, (logits.shape[0],)) < epsilon
    return jnp.where(explore, random_action, greedy_action)


def uniform_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
) -> Float[Array, " B"]:
    """Sample uniformly from valid actions."""
    uniform_logits = jnp.where(action_mask, 0.0, -jnp.inf)
    return jax.random.categorical(key, uniform_logits)
