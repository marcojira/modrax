# Modrax
A compact, modular reinforcement learning library built with JAX and Flax NNX.

> [!WARNING]
> Modrax is under development and its API may change.

> [!NOTE]
> 🏆 **New Craftax result:** The LSTM agent in [`ppo_recurrent_craftax.py`](examples/ppo_recurrent_craftax.py) achieves **19.5%** on **Craftax-1B**.

## Design
Modrax is based around three main object.

- `Env` provides standardized batched environment interaction for different environment suites.
- `Network` is an NNX model that contains the train/evaluation policy
- `Alg` is initialized with an `Env` and a `Network`, then defines a training step that updates the network given interactions with the `Env`.

These three are brought together by `training.py` that contains a training loop which handles logging, checkpointing and evaluation.

## Installation and quick start

Modrax requires Python 3.11 or newer and uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/marcojira/modrax.git
cd modrax

# Install Modrax with the Gymnax environment integration
uv sync --python 3.11 --extra gymnax

# Check the available configuration options
uv run examples/pqn_minatar.py --help
```

Environment integrations are optional. Use `--extra all` to install every available integration. The base dependencies currently install CUDA-enabled JAX.

See the [`examples`](examples) folder for more.

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
