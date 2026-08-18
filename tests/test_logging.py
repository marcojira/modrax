from types import SimpleNamespace

import modrax.logging as logging


def test_disabled_wandb_does_not_render_trajectories(monkeypatch):
    def fail_if_rendered(*_args, **_kwargs):
        raise AssertionError("Rendered trajectories with W&B disabled")

    monkeypatch.setattr(logging, "render_trajectories", fail_if_rendered)
    config = SimpleNamespace(
        save_gif_local=False,
        save_gif_wandb=True,
        wandb=SimpleNamespace(enabled=False),
    )

    logging.log_trajectories(object(), object(), config, epoch=0)
