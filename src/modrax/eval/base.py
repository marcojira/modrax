"""Base evaluation types and functions."""

from typing import Any, Callable

from jaxtyping import Array, Key

from modrax.network.base import Network
from modrax.rollout.base import RolloutData

EvalFn = Callable[[Network, RolloutData, Key[Array, ""]], dict[str, float]]
