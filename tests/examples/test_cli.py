import subprocess
from pathlib import Path

import pytest

repo_root = Path(__file__).parent.parent.parent
examples = list(
    str(example.relative_to(repo_root)) for example in (repo_root / "examples").rglob("*.py")
)


@pytest.mark.parametrize(
    "example",
    [
        pytest.param(example, marks=pytest.mark.xfail(reason="TODO: Broken example"))
        if example
        in [
            "examples/pqn/gtrxl_craftax.py",
            "examples/pqn/rnn_craftax.py",
            "examples/template.py",
        ]
        else example
        for example in examples
    ],
)
def test_examples_cli(example: str):
    """Test that the examples can be imported without errors and that they accept command-line arguments."""
    help_text = subprocess.check_output(f"uv run {example} --help", shell=True)
    assert help_text
