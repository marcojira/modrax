"""Wrapper for the Sequax sequence design environments from https://github.com/marcojira/sequax"""

from dataclasses import dataclass
from typing import Literal

import jax
from jaxtyping import Array, Float, Key
from sequax import AMPSequence, BitSequence, GFPSequence, UTRSequence

from modrax.env.base import (
    DiscreteActionSpec,
    Env,
    EnvConfig,
    State,
    StateWithMetrics,
    StepOutput,
)

TASKS = {
    "BitSequence": BitSequence,
    "AMP": AMPSequence,
    "GFP": GFPSequence,
    "UTR": UTRSequence,
}


@dataclass(frozen=True)
class SequaxConfig(EnvConfig):
    env_name: Literal["BitSequence", "AMP", "GFP", "UTR"] = "BitSequence"
    min_length: int | None = None  # Only AMP has a variable minimum length


class SequaxEnv(Env):
    def __init__(self, config: SequaxConfig):
        kwargs = {} if config.min_length is None else {"min_length": config.min_length}

        # The biological tasks load their bundled reward proxy checkpoint here
        self._env = TASKS[config.env_name](**kwargs)

        # Observations are the token sequence built so far, padded to a fixed length
        self.obs_shape = self._env.obs_shape
        self.pad_token = self._env.pad_token_id  # Networks need it to mask padded positions

        # Every token is an action, so the action count is also the vocabulary size
        self.action_spec = DiscreteActionSpec(self._env.num_actions)

        super().__init__(config)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        sequax_state = self._env.reset(key)

        return State(
            env_state=sequax_state,
            obs=sequax_state.tokens,
            action_mask=self._env.action_mask(sequax_state),
            info={"sequence_length": sequax_state.length},
        )

    def _inner_step_fn(
        self, state: State, action: Array, key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        sequax_output, sequax_state = self._env.step(state.env_state, action, key)

        info = {"sequence_length": sequax_state.length}
        step_output = StepOutput(
            reward=sequax_output.reward,
            done=sequax_output.done,
            truncation=sequax_output.truncation,
            info=info,
        )
        new_state = State(
            env_state=sequax_state,
            obs=sequax_state.tokens,
            action_mask=self._env.action_mask(sequax_state),
            info=info,
        )

        return step_output, new_state

    def terminal_reward(self, state: StateWithMetrics) -> Float[Array, " B"]:
        """Score finished sequences; sequax scores episodes here rather than in step()."""
        return jax.vmap(self._env.terminal_reward)(state.env_state)
