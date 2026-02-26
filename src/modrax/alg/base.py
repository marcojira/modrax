from abc import ABC, abstractmethod

from jaxtyping import Array, Key

from modrax.env.base import Env
from modrax.network.base import Network
from modrax.optimizer import Optimizer


class Alg(ABC):
    total_steps: int  # Subclasses must set in __init__
    env_steps_per_epoch: int

    def __init__(
        self,
        env: Env,
        network: Network,
        optimizer: Optimizer,
        cfg,
        key: Key[Array, ""],
        jit: bool = False,
    ):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.cfg = cfg
        self.jit = jit

    @abstractmethod
    def __call__(self, key: Key[Array, ""]) -> dict[str, float]:
        pass

    @abstractmethod
    def eval(self, key: Key[Array, ""]) -> dict[str, float]:
        pass
