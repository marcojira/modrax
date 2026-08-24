import shlex
import sys
from dataclasses import dataclass

import pytest

from modrax.cli import add_cli


@dataclass(frozen=True)
class SomeConfig:
    learning_rate: float = 3e-4


@add_cli
def some_fn(cfg: SomeConfig):
    return cfg


@pytest.mark.parametrize(
    ("command_line_args", "input", "expected"),
    [
        ("", SomeConfig(learning_rate=0.02), SomeConfig(learning_rate=0.02)),
        ("--learning_rate 0.01", None, SomeConfig(learning_rate=0.01)),
    ],
)
def test_add_cli(
    command_line_args: str,
    input: SomeConfig | None,
    expected: SomeConfig,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "argv", ["some_fn"] + shlex.split(command_line_args))

    if input is None:
        result = some_fn()
    else:
        result = some_fn(input)
    assert result == expected
