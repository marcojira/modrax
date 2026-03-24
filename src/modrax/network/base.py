from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import orbax.checkpoint as ocp
from flax import nnx
from jaxtyping import Array, Float, Int, Key

from modrax.env.base import StateWithMetrics
from modrax.types import Config


@dataclass(frozen=True)
class NetworkConfig(Config):
    pass


class Network(nnx.Module):
    """Base class for all network types. Enables loading/saving functionality"""

    def policy(
        self, env_state: StateWithMetrics, key: Key[Array, ""]
    ) -> tuple[Int[Array, " B"], Any]:
        """Select actions given the current env state. Returns (actions, network_output)
        where network_output contains algorithm-specific data stored during rollout collection."""
        raise NotImplementedError

    def reset(self, done: Float[Array, " B"]):
        return

    def get_carry(self):
        return None

    def save(self, checkpoint_dir_path: str) -> None:
        """Save network to checkpoint directory.

        Args:
            checkpoint_dir_path: Path to directory where the checkpoint will be saved.
            Note that checkpoints are saved in directory (which contains all necessary files)
            instead of a single file (e.g. checkpoint.pkl)
        """
        # Orbax needs absolute path
        checkpoint_dir_path = os.path.abspath(checkpoint_dir_path)
        os.makedirs(checkpoint_dir_path, exist_ok=True)

        # Split the model into graph definition and state
        _, state = nnx.split(self)

        # Save the state (orbax compatible)
        checkpointer = ocp.StandardCheckpointer()
        params_path = os.path.join(checkpoint_dir_path, "params")
        checkpointer.save(params_path, state)
        checkpointer.wait_until_finished()

    def load_checkpoint(self, checkpoint_path: str) -> Network:
        """Load network from checkpoint directory."""
        # Orbax needs absolute path
        checkpoint_path = os.path.abspath(checkpoint_path)

        # TODO: potentially better to use abstract state here (see https://flax.readthedocs.io/en/latest/guides/checkpointing.html)
        graphdef, state = nnx.split(self)

        # Restore the actual state using orbax
        checkpointer = ocp.StandardCheckpointer()
        restored_state = checkpointer.restore(os.path.join(checkpoint_path, "params"), state)

        return nnx.merge(graphdef, restored_state)
