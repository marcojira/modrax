from abc import ABC, abstractmethod
from typing import Any

from flax import nnx
from jaxtyping import Array, Key

from modrax.env.base import Env
from modrax.network.base import Network
from modrax.optimizer import Optimizer


class Alg(ABC):
    total_steps: int  # Subclasses must set in __init__
    env_steps_per_epoch: int
    state: Any

    def __init__(self, env: Env, network: Network, optimizer: Optimizer, cfg):
        self.env = env
        self.network = network
        self.optimizer = optimizer
        self.cfg = cfg

    @abstractmethod
    def __call__(self, key: Key[Array, ""]) -> dict[str, float]:
        pass

    def get_network(self) -> Network:
        """Return the network from the current algorithm state."""
        network, _ = nnx.merge(*self.state.agent_state)
        return network
