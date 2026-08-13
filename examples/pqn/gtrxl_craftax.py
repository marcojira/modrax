"""Train PQN with GTrXL on Craftax."""

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import nagano as ngn
from flax import nnx
from jaxtyping import Array, Float, Int, Key

from modrax.alg.pqn import PQNAlg, PQNConfig, PQNNetwork, PQNNetworkOutput, compute_total_updates
from modrax.env.base import StateWithMetrics
from modrax.env.craftax import CraftaxConfig, CraftaxEnv
from modrax.network.gtrxl import GatedTransformerXL, GTrXLRecurrentState
from modrax.optimizer import Optimizer, OptimizerConfig
from modrax.policy import epsilon_greedy_policy
from modrax.training import TrainConfig, WandbConfig, train
from modrax.types import Config, Shape


@dataclass(frozen=True)
class CraftaxGTrXLNetworkConfig(Config):
    hidden_size: int = 512
    num_layers: int = 1
    norm_type: str = "layer_norm"  # "layer_norm" | "batch_norm" | "none"
    norm_input: bool = True
    add_last_action: bool = True
    # GTrXL-specific
    num_heads: int = 8
    num_transformer_layers: int = 1
    gating: bool = True
    gating_bias: float = 2.0
    segment_len: int = 64
    rollout_memory_len: int = 128


@dataclass(frozen=True)
class CraftaxGTrXLPQNConfig(TrainConfig):
    env_cfg: CraftaxConfig = CraftaxConfig(
        env_name="Craftax-Symbolic-v1",
        optimistic_reset=True,
        num_reset_envs=16,
    )
    network_cfg: CraftaxGTrXLNetworkConfig = CraftaxGTrXLNetworkConfig()
    optimizer_cfg: OptimizerConfig = OptimizerConfig(
        optimizer_type="radam",
        learning_rate=3e-4,
        gradient_clip=0.5,
        lr_decay=True,
    )
    alg_cfg: PQNConfig = PQNConfig(
        total_steps=int(1e9),
        num_envs=1024,
        num_timesteps=128,
        num_minibatches=4,
        num_updates=4,
        gamma=0.99,
        lambd=0.5,
        start_eps=1.0,
        end_eps=0.005,
        eps_decay=0.1,
    )
    wandb: WandbConfig = WandbConfig(enabled=True, project="modrax")
    eval_interval: int = 100
    seed: int = 0
    save_gif_wandb: bool = True
    num_gif_trajectories: int = 2


class CraftaxGTrXLNetwork(PQNNetwork):
    def __init__(
        self,
        obs_shape: Shape,
        action_size: int,
        num_envs: int,
        cfg: CraftaxGTrXLNetworkConfig,
        rngs: nnx.Rngs,
    ):
        self.is_recurrent = True
        self.cfg = cfg
        self.eps = 1.0
        self.action_size = action_size
        obs_dim = 1
        for d in obs_shape:
            obs_dim *= d

        # Input normalization
        self.input_norm = nnx.BatchNorm(obs_dim, rngs=rngs) if cfg.norm_input else None

        # Dense layers before transformer
        self.dense_layers = []
        self.norms = []
        in_dim = obs_dim
        for _ in range(cfg.num_layers):
            self.dense_layers.append(nnx.Linear(in_dim, cfg.hidden_size, rngs=rngs))
            if cfg.norm_type == "layer_norm":
                self.norms.append(nnx.LayerNorm(cfg.hidden_size, rngs=rngs))
            elif cfg.norm_type == "batch_norm":
                self.norms.append(nnx.BatchNorm(cfg.hidden_size, rngs=rngs))
            else:
                self.norms.append(None)
            in_dim = cfg.hidden_size

        # Project to encoder_dim (GTrXL input), including last action if enabled
        gtrxl_input_dim = cfg.hidden_size + (action_size if cfg.add_last_action else 0)
        self.gtrxl = GatedTransformerXL(
            input_dim=gtrxl_input_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_transformer_layers,
            rollout_memory_len=cfg.rollout_memory_len,
            segment_len=cfg.segment_len,
            rngs=rngs,
            gating=cfg.gating,
            gating_bias=cfg.gating_bias,
        )
        self.gtrxl.initialize_carry(num_envs)

        # Q-value output
        self.output = nnx.Linear(gtrxl_input_dim, action_size, rngs=rngs)

        # Track last action
        self.last_action = nnx.Variable(jnp.zeros(num_envs, dtype=jnp.int32))

    def _apply_norm(self, x, norm, train: bool):
        if norm is None:
            return x
        if isinstance(norm, nnx.BatchNorm):
            return norm(x, use_running_average=not train)
        return norm(x)

    def _encode(self, x: Float[Array, "... D"], train: bool = True) -> Float[Array, "... H"]:
        """Encode obs through input norm + dense layers."""
        if self.input_norm is not None:
            x = self.input_norm(x, use_running_average=not train)

        for dense, norm in zip(self.dense_layers, self.norms):
            x = jax.nn.relu(self._apply_norm(dense(x), norm, train))
        return x

    def _concat_last_action(
        self, x: Float[Array, "... H"], last_action: Int[Array, "..."]
    ) -> Float[Array, "... H+A"]:
        """Concat one-hot last action to encoded features."""
        if not self.cfg.add_last_action:
            return x
        one_hot = jax.nn.one_hot(last_action, self.action_size)
        return jnp.concatenate([x, one_hot], axis=-1)

    def policy(self, env_state: StateWithMetrics, key: Key[Array, ""]):
        obs = env_state.obs.reshape(env_state.obs.shape[0], -1)
        encoded = self._encode(obs, train=False)
        encoded = self._concat_last_action(encoded, self.last_action.value)

        carry, out = self.gtrxl(encoded)
        q_values = self.output(out)

        action = epsilon_greedy_policy(q_values, env_state.action_mask, key, self.eps)
        self.last_action.value = action
        return action, PQNNetworkOutput(q_values, carry=(carry, action))

    def train_forward(
        self,
        obs: Float[Array, "B T ..."],
        dones: Float[Array, "B T"],
        init_carry,
        saved_carry,
    ):
        B, T = obs.shape[:2]

        # Unpack carry: (gtrxl_state, last_action) tuples
        gtrxl_init_carry, initial_last_action = init_carry
        gtrxl_saved_carry, saved_actions = saved_carry

        # Encode observations
        flat_obs = obs.reshape(B, T, -1)
        encoded = self._encode(flat_obs)

        # Reconstruct last_actions
        last_actions = jnp.concatenate(
            [initial_last_action[:, None], saved_actions[:, :-1]], axis=1
        )
        encoded = self._concat_last_action(encoded, last_actions)

        # GTrXL forward (uses its own memory mechanism, not dones)
        out = self.gtrxl.train_forward(encoded, gtrxl_init_carry, gtrxl_saved_carry)
        return self.output(out)

    def reset(self, done: Float[Array, " B"]):
        self.gtrxl.reset(done)
        self.last_action.value = jnp.where(done, 0, self.last_action.value)

    def get_carry(self):
        return (self.gtrxl.carry.value, self.last_action.value)


def main(cfg):
    key = jax.random.key(cfg.seed)

    # Init objects
    env = CraftaxEnv(cfg.env_cfg)
    network = CraftaxGTrXLNetwork(
        env.obs_shape, env.action_size, cfg.alg_cfg.num_envs, cfg.network_cfg, nnx.Rngs(cfg.seed)
    )
    optimizer = Optimizer(cfg.optimizer_cfg, network, compute_total_updates(cfg.alg_cfg))
    alg = PQNAlg(env, network, optimizer, cfg.alg_cfg, key=key)

    train(alg, cfg)


if __name__ == "__main__":
    project = ngn.init("modrax")

    gpu = ngn.GPU.L40S
    time = "12:00:00"
    slurm_cfg = ngn.SlurmConfig(
        partition=ngn.Partition.LONG, gpu=gpu, num_cpus=4, ram_gb=24, time=time
    )

    def fn(num_transformer_layers, hidden_size, **kwargs):
        cfg = CraftaxGTrXLPQNConfig()
        # cfg.network_cfg.num_transformer_layers = num_transformer_layers
        # cfg.network_cfg.hidden_size = hidden_size
        cfg = dataclasses.replace(
            cfg,
            network_cfg=dataclasses.replace(
                cfg.network_cfg, num_transformer_layers=num_transformer_layers, hidden_size=hidden_size
            ),
        )
        main(cfg)

    project.run_exp(
        "pqn/gtrxl_craftax",
        fn,
        {"num_transformer_layers": [1], "hidden_size": [256, 512]},
        slurm_cfg,
        extra_commands=[
            "source /home/mila/m/marco.jiralerspong/projects/modrax/.venv/bin/activate"
        ],
        num_workers=2,
    )
