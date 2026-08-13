"""Wrapper for PGX Minatar envs, from https://github.com/sotetsuk/pgx"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Key
from pgx import make
from pgx.minatar.asterix import MinAtarAsterix
from pgx.minatar.breakout import MinAtarBreakout
from pgx.minatar.freeway import MinAtarFreeway
from pgx.minatar.seaquest import MinAtarSeaquest
from pgx.minatar.space_invaders import MinAtarSpaceInvaders

from modrax.env.base import DiscreteActionSpec, Env, EnvConfig, State, StateWithMetrics, StepOutput

MINATAR_ENV_MAP = {
    "minatar-asterix": MinAtarAsterix,
    "minatar-breakout": MinAtarBreakout,
    "minatar-freeway": MinAtarFreeway,
    "minatar-seaquest": MinAtarSeaquest,
    "minatar-space_invaders": MinAtarSpaceInvaders,
}


@dataclass(frozen=True)
class PGXConfig(EnvConfig):
    env_name: Literal[
        "2048",
        "animal_shogi",
        "backgammon",
        "bridge_bidding",
        "chess",
        "connect_four",
        "gardner_chess",
        "go_9x9",
        "go_19x19",
        "hex",
        "kuhn_poker",
        "leduc_holdem",
        "minatar-asterix",
        "minatar-breakout",
        "minatar-freeway",
        "minatar-seaquest",
        "minatar-space_invaders",
        "othello",
        "shogi",
        "sparrow_mahjong",
        "tic_tac_toe",
    ] = "minatar-asterix"
    sticky_action_prob: float = 0.1


class PGXEnv(Env):
    def __init__(self, config: PGXConfig):
        if config.env_name in MINATAR_ENV_MAP:
            env_cls = MINATAR_ENV_MAP[config.env_name]
            self._env = env_cls(sticky_action_prob=config.sticky_action_prob)
        else:
            self._env = make(config.env_name)

        self.obs_shape = self._env.observation_shape
        self.action_spec = DiscreteActionSpec(self._env.num_actions)

        super().__init__(config)

    def _inner_reset_fn(self, key: Key[Array, ""]) -> State:
        pgx_state = self._env.init(key)
        return State(
            env_state=pgx_state,
            obs=pgx_state.observation,
            action_mask=pgx_state.legal_action_mask.astype(jnp.bool),
            info={"current_player": pgx_state.current_player, "turn": 0},
        )

    def _inner_step_fn(
        self, state: State, action: Array, key: Key[Array, ""]
    ) -> tuple[StepOutput, State]:
        step = state.env_state._step_count
        pgx_state = self._env.step(state.env_state, action, key)

        if self.config.env_name in MINATAR_ENV_MAP:
            reward = jnp.squeeze(pgx_state.rewards, -1)
        else:
            # By default, pgx rewards are relative to current_player which is different from
            # the player who's turn it is. We modify this to always return the reward of the player
            # whose turn it is
            player_turn = step % 2

            # Whether current_player is aligned with the player whose turn it is
            should_flip = pgx_state.current_player != (pgx_state._step_count % 2)

            reward = jnp.where(
                should_flip,
                -pgx_state.rewards[player_turn],
                pgx_state.rewards[player_turn],
            )

        done = pgx_state.terminated

        step_output = StepOutput(
            reward=reward, done=done.astype(jnp.bool), truncation=jnp.bool_(False), info={}
        )
        new_state = State(
            env_state=pgx_state,
            obs=pgx_state.observation,
            action_mask=pgx_state.legal_action_mask.astype(jnp.bool),
            info={"current_player": pgx_state.current_player, "turn": pgx_state._step_count},
        )

        return step_output, new_state

    @staticmethod
    def _get_minatar_cmap(n_channels: int):
        """Get colormap for MinAtar visualization."""
        assert n_channels in (4, 6, 7, 10)
        if n_channels == 4:
            return [
                (0.08605633600581405, 0.23824692404212, 0.30561236308077167),
                (0.32927729263408284, 0.4762845556584382, 0.1837155549758328),
                (0.8146245329198283, 0.49548316572322215, 0.5752525936416857),
                (0.7587183008012618, 0.7922069335474338, 0.9543861221913403),
            ]
        elif n_channels == 6:
            return [
                (0.10231025194333628, 0.13952898866828906, 0.2560120319409181),
                (0.10594361078604106, 0.3809739011595331, 0.27015111282899046),
                (0.4106130272672762, 0.48044780541672255, 0.1891154277778484),
                (0.7829183382530567, 0.48158303462490826, 0.48672451968362596),
                (0.8046168329276406, 0.6365733569301846, 0.8796578402926125),
                (0.7775608374378459, 0.8840392521212448, 0.9452007992345052),
            ]
        elif n_channels == 7:
            return [
                (0.10419418740482515, 0.11632019220053316, 0.2327552016195138),
                (0.08523511613408935, 0.32661779003565533, 0.2973201282529313),
                (0.26538761550634205, 0.4675654910052002, 0.1908220644759285),
                (0.6328422475018423, 0.4747981096220677, 0.29070209208025455),
                (0.8306875710682655, 0.5175161303658079, 0.6628221028832032),
                (0.7779565181455343, 0.7069421942599752, 0.9314406084043191),
                (0.7964528047840354, 0.908668973545918, 0.9398253500983916),
            ]
        elif n_channels == 10:
            return [
                (0.09854228363950114, 0.07115215572295082, 0.16957891809124037),
                (0.09159726558869188, 0.20394337960213008, 0.29623965888210324),
                (0.09406611799930162, 0.3578871412608098, 0.2837709711722866),
                (0.23627685553553793, 0.46114369021199075, 0.19770731888985724),
                (0.49498740849493095, 0.4799034869159042, 0.21147789468974837),
                (0.7354526513473981, 0.4748861903571046, 0.40254094042448907),
                (0.8325928529853291, 0.5253446757844744, 0.6869376931865354),
                (0.7936920632275369, 0.6641337211433709, 0.9042311843062529),
                (0.7588424692372241, 0.8253990353420474, 0.9542699331220588),
                (0.8385645211683802, 0.9411869386771845, 0.9357655639413166),
            ]

    def render(self, state: State | StateWithMetrics) -> np.ndarray:
        """Render MinAtar environments from PGX."""
        # Modified from https://github.com/kenjyoung/MinAtar
        obs = state.env_state.observation

        if self.config.env_name in MINATAR_ENV_MAP:
            return self.render_obs(obs)
        else:
            return np.zeros((10, 10, 3))  # TODO: Implement visualization for board games

    def batch_render(self, states: StateWithMetrics) -> np.ndarray:
        if self.config.env_name not in MINATAR_ENV_MAP:
            return np.zeros((states.obs.shape[0], 10, 10, 3), dtype=np.uint8)

        obs = np.array(states.env_state.observation)  # (B, H, W, C)
        n_channels = obs.shape[-1]
        cmap_colors = PGXEnv._get_minatar_cmap(n_channels)
        color_table = (np.array([(0, 0, 0)] + cmap_colors) * 255).astype(np.uint8)

        channel_idx = np.max(obs * np.arange(1, n_channels + 1), axis=-1).astype(np.intp)
        return color_table[channel_idx]  # (B, H, W, 3)

    @staticmethod
    def render_obs(obs: Array) -> np.ndarray:
        n_channels = obs.shape[-1]
        cmap_colors = PGXEnv._get_minatar_cmap(n_channels)
        color_table = (np.array([(0, 0, 0)] + cmap_colors) * 255).astype(np.uint8)

        # channel_idx: 0=background, k+1=channel k active
        channel_idx = np.max(np.array(obs) * np.arange(1, n_channels + 1), axis=-1).astype(
            np.intp
        )
        return color_table[channel_idx]  # (H, W, 3)
