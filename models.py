"""Domain models: City and Individual (a candidate tour)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass(frozen=True)
class City:
    """A city as a point on the plane."""

    id: int
    x: float
    y: float

    def distance_to(self, other: "City") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class Individual:
    """A candidate solution: a permutation of city indices (a tour).

    Fitness is cached and invalidated automatically whenever the
    chromosome is replaced via `chromosome.setter`.
    """

    __slots__ = ("_chromosome", "_distance", "_fitness")

    def __init__(self, chromosome: Sequence[int]):
        self._chromosome: List[int] = list(chromosome)
        self._distance: float | None = None
        self._fitness: float | None = None

    @property
    def chromosome(self) -> List[int]:
        return self._chromosome

    @chromosome.setter
    def chromosome(self, value: Sequence[int]) -> None:
        self._chromosome = list(value)
        self._distance = None
        self._fitness = None

    def copy(self) -> "Individual":
        return Individual(self._chromosome)

    def __len__(self) -> int:
        return len(self._chromosome)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Individual(distance={self._distance}, fitness={self._fitness})"
