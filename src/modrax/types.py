from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

Shape: TypeAlias = Sequence[int]


@dataclass(frozen=True)
class Config:
    pass
