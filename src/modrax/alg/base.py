from abc import ABC, abstractmethod

from jaxtyping import Array, Key
from pydantic import BaseModel

from modrax.env.base import Env
from modrax.network.base import Network
from modrax.optimizer import Optimizer


class AlgConfig(BaseModel):
    model_config = {"frozen": True}
    num_gen_steps: int


class Alg(ABC):
    def __init__(
        self,
        env: Env,
        network: Network,
        optimizer: Optimizer,
        alg_config: AlgConfig,
        key: Key[Array, ""],
        jit: bool = False,
    ):
        self.env_steps_per_epoch = 0  # Set during init
        pass

    @abstractmethod
    def __call__(self, key: Key[Array, ""]) -> dict[str, float]:
        pass
