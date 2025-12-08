from typing import NamedTuple, TypedDict

from flax import struct
from jaxtyping import Array, Float


class NetworkInput(TypedDict):
    obs: Float[Array, "B ..."]


class NetworkOutput(TypedDict):
    policy: Float[Array, "B A"]


class Trajectory(NamedTuple):
    obs: Float[Array, "T B ..."]
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
