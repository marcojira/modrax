# ModRax

A modular JAX reinforcement learning framework with composable components.

## Overview

ModRax is designed to simplify reinforcement learning research with JAX environments. Built on the Flax NNX API, it provides a clean, modular architecture that serves as a starting point for RL research and experimentation.

## Installation

Using [uv](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -e ".[dev]"
```

Or using pip:

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- JAX and Flax (NNX API)

## Project Structure

ModRax follows a modular architecture where:
- Each module folder corresponds to an abstract base class
- Base classes are defined in `__init__.py` for clean imports
- Implementations are in separate files within each module folder

## License

MIT
