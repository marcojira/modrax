"""Base classes for network blocks.

Blocks are reusable building blocks.
Blocks all have the same __init__ args
- input_shape
- output_dim
- config
- rngs
They are meant to be composed into networks.
"""

from abc import ABC, abstractmethod

from flax import nnx
from jaxtyping import Array, Float
from pydantic import BaseModel

from modrax.types import Shape


class BlockConfig(BaseModel):
    """Base configuration for blocks."""

    pass


class BlockBase(nnx.Module, ABC):
    """Abstract base class for all network blocks.

    All blocks must implement:
    - __init__(input_shape, output_dim, config, rngs)

    The __call__ signature is specified by block type subclasses.
    """

    @abstractmethod
    def __init__(
        self,
        input_shape: Shape | int,
        output_dim: int,
        config: BlockConfig,
        rngs: nnx.Rngs,
    ):
        pass


class Block(BlockBase, ABC):
    """Abstract base class for STANDARD blocks which trnasform inputs to outputs.

    Can be used as encoders, trunks, or heads.
    Examples: MLP, Linear, CNN
    """

    @abstractmethod
    def __call__(self, x: Float[Array, "B ..."]) -> Float[Array, "B D"]:
        """Transform batched input of arbitrary shape to batched output."""
        pass


class RecurrentState:
    """Base class for different types of reccurent states"""

    pass


class RecurrentBlock(BlockBase, ABC):
    """Abstract base class for recurrent blocks.

    Take in an additional recurrent state and output the updated recurrent state.
    Examples: RNN, LSTM, GRU, GTrXL
    """

    @abstractmethod
    def __call__(
        self,
        x: Float[Array, "B T ..."],
        state: RecurrentState,
    ) -> tuple[Float[Array, "B T output_dim"], RecurrentState]:
        """Process sequential input with recurrent state.

        Output both the the output and the updated recurrent state
        """
        pass

    @abstractmethod
    def init_recurrent_state(self, *args, **kwargs) -> RecurrentState:
        """Initialize the recurrent state (with block-specific arguments)"""
        pass

    @abstractmethod
    def reset_recurrent_state(self, *args, **kwargs) -> RecurrentState:
        """Reset the recurrent state (with block-specific arguments)"""
        pass

    @abstractmethod
    def train_forward(
        self,
        encoded_obs: Float[Array, "B T D"],
        recurrent_state: RecurrentState,
        init_recurrent_state: RecurrentState,
    ) -> dict[str, Array]:
        pass
