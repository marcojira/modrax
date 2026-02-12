from abc import abstractmethod

from flax import nnx
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
    def __call__(self, iteration_key) -> dict[str, float]:
        pass

    def get_network(self):
        return nnx.merge(*self.state.network_state)
