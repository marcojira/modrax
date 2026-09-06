# Modrax
A concise, extandable and performant RL library based on Jax and NNX. The goal is to be somewhere between the one-file implementations with duplicated code and large complex frameworks where customization is hard.

## Design

Modrax is organized around three composable objects:

- **Environment** wrapper over common RL suites (`gymnax`, `pgx`, `octax`, `craftax` and `mujoco_playground`) given them a common interface.
  - Also additional functionality like auto_reset, optimistic resets, etc.
- **Network** checkpointable support for NNX networks.
- **Algorithm** takes a network and an environment and implements a training step function that updates the network.

These three are brought together by `training.py` that implements a training loop and handles logging, checkpointing and evaluation.

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

