from typing import Protocol

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Key


class PolicyFn(Protocol):
    def __call__(
        self,
        logits: Float[Array, "B A"],
        action_mask: Float[Array, "B A"],
        key: Key[Array, ""],
    ) -> Float[Array, "B ..."]: ...


def softmax_policy(
    logits: Float[Array, "B A"],
    action_mask: Float[Array, "B A"],
    key: Key[Array, ""],
) -> Float[Array, " B"]:
    masked_logits = jnp.where(action_mask, logits, -jnp.inf)
    action = jax.random.categorical(key, masked_logits)

    return action
