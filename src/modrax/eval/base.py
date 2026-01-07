"""Base evaluation types and functions."""

from typing import Protocol

from jaxtyping import Array, Key
from pydantic import BaseModel

from modrax.env.base import Env
from modrax.network.base import Network


class EvalConfig(BaseModel):
    model_config = {"frozen": True}


class EvalFn(Protocol):
    def __call__(
        self,
        network: Network,
        env: Env,
        config: EvalConfig,
        key: Key[Array, ""],
    ) -> dict[str, float]: ...
