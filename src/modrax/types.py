from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

Shape: TypeAlias = Sequence[int]


@dataclass
class Config:
    pass
