from abc import abstractmethod
from typing import Any, Callable

from pgx import Env
from pydantic import BaseModel

from modrax.env.base import EnvState
from modrax.network.base import Network
from modrax.network.block.base import RecurrentState
from modrax.optimizer import Optimizer


class AlgConfig(BaseModel):
    model_config = {"frozen": True}
    num_gen_steps: int


class Alg:
    def __init__(
        self,
        env_state: EnvState,
        recurrent_state: RecurrentState | None,
        network: Network,
        optimizer: Optimizer,
        env: Env,
        alg_config: AlgConfig,
        jit: bool = False,
    ):
        pass

    @abstractmethod
    def __call__(
        self, network, optimizer, iteration_key
    ) -> tuple[Network, Optimizer, dict[str, float]]:
        pass
