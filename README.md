# Modrax
An easy-to-use, performant RL library based on Jax and NNX.

## Design

Modrax is organized around three explicit, composable objects:

- **Environment** defines interaction, observations, and the action space.
- **Network** defines the policy or value model and owns its learned state.
- **Algorithm** combines the environment and network, implements the learning procedure, and owns
  its optimization strategy and state.

Keeping these responsibilities separate makes each object independently replaceable and keeps
experiment setup explicit.

## Quick start
```bash
git clone https://github.com/marcojira/modrax.git
cd modrax
git checkout dev
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .

# Launch example script
python examples/pqn/minatar.py
```
