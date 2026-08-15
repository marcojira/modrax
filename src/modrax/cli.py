"""Command-line interface helpers."""

import typing
from typing import Callable

Out = typing.TypeVar("Out")
ConfigType = typing.TypeVar("ConfigType")


def add_cli(fn: Callable[[ConfigType], Out]):
    """Add a simple command-line interface to a function accepting a dataclass."""
    import rich_argparse
    import simple_parsing

    class _FormatterClass(rich_argparse.RichHelpFormatter, simple_parsing.SimpleHelpFormatter): ...  # type: ignore

    def wrapper(cfg: ConfigType | None = None) -> Out:
        if not cfg:
            config_type = typing.get_type_hints(fn).popitem()[1]
            cfg = simple_parsing.parse(config_type, formatter_class=_FormatterClass, description=fn.__doc__)
            assert cfg
        return fn(cfg)

    return wrapper
