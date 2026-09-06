# Modrax
A concise, extandable and performant RL library based on Jax and NNX. The goal is to reduce code duplication and ad hoc
structure while hopefully remaining legible and easy to customize.

> [!IMPORTANT]
> 🏆 **New Craftax result:** The LSTM agent in [`ppo_recurrent_craftax.py`](examples/ppo_recurrent_craftax.py) achieves **19.5%** on **Craftax-1B**.

## Design

Modrax is organized around three composable objects. Each has a base class that defines its common interface:

- `Environment`: Common interface over RL suites, exposes `obs_shape`, `action_spec`, and batched `reset` and `step` methods
- `Network`: NNX network, exposes `policy` and `eval_policy`, with hooks for resetting recurrent state.
  - Also provides easy to use checkpointing methods.
- `Algorithm`: Initialized with an `Environment` and a `Network`. Must implement a `step` method that runs one training
  iteration and returns training metrics.

These three are brought together by `training.py` that contains a training loop and handles logging, checkpointing and evaluation.

## Quick start
```bash
git clone https://github.com/marcojira/modrax.git
cd modrax

# Install with envs you need: craftax, gymnax, octax, pgx, mujoco (or `all` for every one)
uv sync --extra gymnax

# Launch example script
uv run examples/pqn_minatar.py
```

We recommend checking out the `examples` folder to get a better understanding of Modrax's structure and how to use it!

[WIP 🚧] More documentation coming soon.

## Environment

## Network

## Algorithm

## Training

## Functions

### Buffer

### Eval

### Policy

### Rollout
