"""Abstract strategy interfaces.

`GeneticAlgorithm` depends only on these abstractions (Dependency
Inversion). Each is narrow and single-purpose (Interface
Segregation), so a concrete class only implements what it needs.
Adding a brand-new selection/crossover/mutation/elitism scheme never
requires modifying `genetic_algorithm.py` (Open/Closed).
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Sequence

from .models import City, Individual


class SelectionStrategy(ABC):
    """Chooses parents from a population for reproduction."""

    @abstractmethod
    def select(
        self,
        population: Sequence[Individual],
        cities: Sequence[City],
        num_parents: int,
        rng: random.Random,
    ) -> List[Individual]:
        """Return `num_parents` individuals (with repetition allowed)."""
        raise NotImplementedError


class CrossoverStrategy(ABC):
    """Combines two parents into a child chromosome.

    Order-based crossover (not bit/value crossover as in a simple GA)
    is required here: a TSP chromosome is a *permutation*, so naive
    single-point crossover would produce invalid tours with repeated
    or missing cities. This is the key encoding/crossover difference
    from a simple GA that this lab asks students to internalize.
    """

    @abstractmethod
    def crossover(self, parent1: Individual, parent2: Individual, rng: random.Random) -> Individual:
        raise NotImplementedError


class MutationStrategy(ABC):
    """Perturbs a chromosome while keeping it a valid permutation."""

    @abstractmethod
    def mutate(self, individual: Individual, mutation_rate: float, rng: random.Random) -> Individual:
        raise NotImplementedError


class ElitismStrategy(ABC):
    """Selects individuals to carry over unchanged into the next generation."""

    @abstractmethod
    def select_elite(self, population: Sequence[Individual], cities: Sequence[City]) -> List[Individual]:
        """Return the elite (possibly empty list -> no elitism)."""
        raise NotImplementedError

    @property
    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Island-model strategies
# ---------------------------------------------------------------------------


class MigrationTopology(ABC):
    """Defines *where* migrants travel: which islands each island sends to."""

    @abstractmethod
    def targets(self, island_index: int, num_islands: int, rng: random.Random) -> List[int]:
        """Return indices of the islands that receive migrants from `island_index`."""
        raise NotImplementedError


class MigrantSelectionStrategy(ABC):
    """Defines *who* leaves an island (which individuals become migrants)."""

    @abstractmethod
    def select_migrants(
        self,
        population: Sequence[Individual],
        cities: Sequence[City],
        count: int,
        rng: random.Random,
    ) -> List[Individual]:
        """Return `count` individuals to send (the originals stay on the island)."""
        raise NotImplementedError


class ReplacementStrategy(ABC):
    """Defines *whom* incoming migrants replace on the receiving island."""

    @abstractmethod
    def replace(
        self,
        population: Sequence[Individual],
        incoming: Sequence[Individual],
        cities: Sequence[City],
        rng: random.Random,
    ) -> List[Individual]:
        """Return a new population of the same size with `incoming` inserted."""
        raise NotImplementedError
