from types import SimpleNamespace

import jax.numpy as jnp
from flax import nnx

import modrax.training as training
from modrax.network import Network


class _Network(Network):
    def __init__(self):
        self.weight = nnx.Param(jnp.zeros(()))


class _Alg:
    def __init__(self):
        self.env = object()
        self.network = _Network()
        self.total_steps = 1
        self.env_steps_per_epoch = 1
        self.trajectories = object()

    def __call__(self, key):
        return {"loss": 0.0}

def test_train_uses_algorithm_dependencies(monkeypatch):
    alg = _Alg()
    rendered = []
    monkeypatch.setattr(
        training,
        "evaluate",
        lambda algorithm, *args, **kwargs: ({"return": 0.0}, algorithm.trajectories),
    )
    monkeypatch.setattr(
        training,
        "log_trajectories",
        lambda env, trajectories, *args, **kwargs: rendered.append((env, trajectories)),
    )
    cfg = SimpleNamespace(
        seed=0,
        eval_interval=1,
        eval_max_steps=10,
        display_network=False,
        save_path=None,
        num_gif_trajectories=1,
        wandb=SimpleNamespace(enabled=False),
    )

    network = training.train(alg, cfg)

    assert network is alg.network
    assert rendered == [(alg.env, alg.trajectories)]
