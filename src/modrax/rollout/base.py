from typing import Any, Callable, NamedTuple, Protocol, TypedDict

from flax import struct
from jaxtyping import Array, Float, Key
from pydantic import BaseModel


class NetworkInput(TypedDict):
    obs: Float[Array, "B ..."]


class NetworkOutput(TypedDict):
    policy: Float[Array, "B A"]


class Trajectory(NamedTuple):
    obs: Float[Array, "T B ..."]
    info: Any
    actions: Float[Array, "T B ..."]
    rewards: Float[Array, "T B"]
    action_masks: Float[Array, "T B A"]
    network_output: NetworkOutput
    dones: Float[Array, "T B"]
    episode_returns: Float[Array, "T B"]
    episode_lengths: Float[Array, "T B"]


@struct.dataclass
class RolloutData:
    trajectory: Trajectory
    final_out: NetworkOutput


class RolloutConfig(BaseModel):
    model_config = {"frozen": True}
    num_steps: int = 128


class RolloutFn(Protocol):
    def __call__(
        self,
        network: Any,
        policy_fn: Callable,
        step_fn: Callable,
        env_state: Any,
        recurrent_state: Any,
        config: RolloutConfig,
        key: Key[Array, ""],
    ) -> tuple[Any, Any | None, RolloutData]: ...
