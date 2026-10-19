"""Elitism strategies.

Elitism is modelled as a strategy in its own right, including an
explicit "do nothing" implementation, so the GA loop can enable or
disable elitism purely through dependency injection - no branching
in `genetic_algorithm.py`.
"""

from __future__ import annotations

from typing import List, Sequence

from .fitness import get_distance
from .interfaces import ElitismStrategy
from .models import City, Individual


class StandardElitism(ElitismStrategy):
    """Carries the `count` best individuals (lowest tour distance) forward unchanged."""

    def __init__(self, count: int = 1):
        if count < 0:
            raise ValueError("count must be >= 0")
        self._count = count

    @property
    def count(self) -> int:
        return self._count

    def select_elite(self, population: Sequence[Individual], cities: Sequence[City]) -> List[Individual]:
        if self._count == 0:
            return []
        ranked = sorted(population, key=lambda ind: get_distance(ind, cities))
        return [ind.copy() for ind in ranked[: self._count]]


class NoElitism(ElitismStrategy):
    """Explicit null-object: no individual is protected from replacement."""

    @property
    def count(self) -> int:
        return 0

    def select_elite(self, population: Sequence[Individual], cities: Sequence[City]) -> List[Individual]:
        return []
