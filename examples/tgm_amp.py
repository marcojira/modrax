"""Train TGM on the Sequax AMP sequence design task."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Bool, Float, Int, Key

from modrax.alg.base import OptimizerConfig
from modrax.alg.tgm import TGMAlg, TGMConfig, TGMNetwork, policy_logits
from modrax.cli import add_cli
from modrax.env.base import StateWithMetrics
from modrax.env.sequax import SequaxConfig, SequaxEnv
from modrax.network import NetworkConfig
from modrax.network.mlp import MLP
from modrax.policy import epsilon_softmax_policy
from modrax.training import TrainConfig, WandbConfig, train

MAX_EPISODE_STEPS = 61  # AMP holds at most 60 residues, and one more step emits EOS


@dataclass(frozen=True)
class SequenceTransformerConfig(NetworkConfig):
    embed_dim: int = 64
    hidden_dim: int = 64
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.1
    eps: float = 0.01  # Probability of sampling a uniformly random token


@dataclass(frozen=True)
class TGMAMPConfig(TrainConfig):
    env_cfg: SequaxConfig = SequaxConfig(env_name="AMP", auto_reset=False, min_length=15)
    network_cfg: SequenceTransformerConfig = SequenceTransformerConfig()
    alg_cfg: TGMConfig = TGMConfig(
        optimizer_cfg=OptimizerConfig(
            optimizer_type="adamw", learning_rate=1e-4, weight_decay=1e-4, gradient_clip=10.0
        ),
        total_steps=MAX_EPISODE_STEPS * 100_000,  # Roughly 100k generated sequences
        num_envs=16,
        num_timesteps=MAX_EPISODE_STEPS,
        alpha=1.0,
        omega=4.0,
        q=0.5,
        reward_exp=64.0,
    )
    wandb: WandbConfig = WandbConfig(enabled=False, project="modrax")
    seed: int = 0


def sinusoidal_encoding(seq_len: int, embed_dim: int) -> Float[Array, "1 T D"]:
    """Fixed sinusoidal positional encoding."""
    positions = jnp.arange(seq_len)[:, None]
    frequencies = jnp.exp(jnp.arange(0, embed_dim, 2) * (-jnp.log(10000.0) / embed_dim))

    encoding = jnp.zeros((seq_len, embed_dim))
    encoding = encoding.at[:, 0::2].set(jnp.sin(positions * frequencies))
    encoding = encoding.at[:, 1::2].set(jnp.cos(positions * frequencies))

    return encoding[None]


class SequenceTransformer(TGMNetwork):
    """Causal transformer that scores the next token at every position of a sequence."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_tokens: int,
        pad_token: int,
        cfg: SequenceTransformerConfig,
        alg_cfg: TGMConfig,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.alg_cfg = alg_cfg  # The sampled policy is the one TGM optimizes
        self.pad_token = pad_token
        self.positional_encoding = sinusoidal_encoding(obs_shape[0], cfg.embed_dim)

        # Unit-scale embeddings, following the reference, so positions don't swamp tokens
        self.embed = nnx.Embed(
            num_tokens,
            cfg.embed_dim,
            embedding_init=nnx.initializers.truncated_normal(1.0),
            rngs=rngs,
        )
        self.dropout = nnx.Dropout(cfg.dropout, rngs=rngs)

        # Following the reference, each head gets a full embed_dim of the key, not a split of it
        self.attention = nnx.List(
            [
                nnx.MultiHeadAttention(
                    num_heads=cfg.num_heads,
                    in_features=cfg.embed_dim,
                    qkv_features=cfg.embed_dim * cfg.num_heads,
                    out_features=cfg.embed_dim,
                    decode=False,
                    kernel_init=nnx.initializers.variance_scaling(
                        2 / cfg.num_layers, "fan_in", "truncated_normal"
                    ),
                    rngs=rngs,
                )
                for _ in range(cfg.num_layers)
            ]
        )
        # Dropout sits between the two layers, so the block is spelled out instead of using MLP
        self.feed_forward_in = nnx.List(
            [nnx.Linear(cfg.embed_dim, cfg.hidden_dim, rngs=rngs) for _ in range(cfg.num_layers)]
        )
        self.feed_forward_out = nnx.List(
            [nnx.Linear(cfg.hidden_dim, cfg.embed_dim, rngs=rngs) for _ in range(cfg.num_layers)]
        )
        self.attention_norms = nnx.List(
            [nnx.LayerNorm(cfg.embed_dim, rngs=rngs) for _ in range(cfg.num_layers)]
        )
        self.feed_forward_norms = nnx.List(
            [nnx.LayerNorm(cfg.embed_dim, rngs=rngs) for _ in range(cfg.num_layers)]
        )

        self.head = MLP(
            cfg.embed_dim,
            (4 * cfg.hidden_dim, 4 * cfg.hidden_dim),
            num_tokens,
            jax.nn.relu,
            rngs=rngs,
        )

    def attention_mask(self, tokens: Int[Array, "B T"]) -> Bool[Array, "B 1 T T"]:
        """Attend to earlier, non-padding positions only."""
        seq_len = tokens.shape[-1]
        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
        return (tokens != self.pad_token)[:, None, None, :] & causal[None, None]

    def train_forward(self, obs: Int[Array, "B T"]) -> Float[Array, "B T A"]:
        mask = self.attention_mask(obs)

        x = self.dropout(self.embed(obs) + self.positional_encoding)
        for attention, norm, ff_in, ff_out, ff_norm in zip(
            self.attention,
            self.attention_norms,
            self.feed_forward_in,
            self.feed_forward_out,
            self.feed_forward_norms,
        ):
            x = norm(x + self.dropout(attention(x, mask=mask)))
            feed_forward = ff_out(self.dropout(jax.nn.relu(ff_in(x))))
            x = ff_norm(x + feed_forward)

        return self.head(x)

    def next_token_logits(self, env_state: StateWithMetrics) -> Float[Array, "B A"]:
        """Score the position the next token is written to."""
        logits = self.train_forward(env_state.obs)
        position = env_state.info["sequence_length"]

        return jnp.take_along_axis(logits, position[:, None, None], axis=1).squeeze(axis=1)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        logits = policy_logits(self.next_token_logits(env_state), self.alg_cfg)
        action = epsilon_softmax_policy(logits, env_state.action_mask, key, self.cfg.eps)

        return action, None

    def eval_policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        logits = policy_logits(self.next_token_logits(env_state), self.alg_cfg)
        action = epsilon_softmax_policy(logits, env_state.action_mask, key, 0.0)

        return action, None


@add_cli
def main(cfg: TGMAMPConfig):
    key = jax.random.key(cfg.seed)
    network_key, alg_key, train_key = jax.random.split(key, 3)

    # Init objects
    env = SequaxEnv(cfg.env_cfg)
    network = SequenceTransformer(
        env.obs_shape,
        env.action_size,
        env.pad_token,
        cfg.network_cfg,
        cfg.alg_cfg,
        nnx.Rngs(network_key),
    )
    alg = TGMAlg(env, network, cfg.alg_cfg, key=alg_key)

    train(alg, cfg, key=train_key)


if __name__ == "__main__":
    main()
