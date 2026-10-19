"""Concrete migration strategies for the island model.

Three independent axes, each its own Strategy (Open/Closed + Interface
Segregation - add a new policy without touching the island driver):

* topology          - where migrants go     (ring / fully connected / random)
* migrant selection - who leaves an island  (best / random)
* replacement       - whom migrants replace (worst / random)
"""

from __future__ import annotations

import random
from typing import List, Sequence

from .fitness import get_distance
from .interfaces import MigrantSelectionStrategy, MigrationTopology, ReplacementStrategy
from .models import City, Individual


# ----------------------------- topologies ---------------------------------


class RingTopology(MigrationTopology):
    """Island i sends to island (i + 1) mod N - the classic one-way ring."""

    def targets(self, island_index: int, num_islands: int, rng: random.Random) -> List[int]:
        if num_islands < 2:
            return []
        return [(island_index + 1) % num_islands]


class FullyConnectedTopology(MigrationTopology):
    """Every island sends to every other island (maximum gene flow)."""

    def targets(self, island_index: int, num_islands: int, rng: random.Random) -> List[int]:
        return [j for j in range(num_islands) if j != island_index]


class RandomTopology(MigrationTopology):
    """Each island sends to one randomly chosen other island per migration."""

    def targets(self, island_index: int, num_islands: int, rng: random.Random) -> List[int]:
        if num_islands < 2:
            return []
        choices = [j for j in range(num_islands) if j != island_index]
        return [rng.choice(choices)]


# ------------------------- migrant selection ------------------------------


class BestMigrants(MigrantSelectionStrategy):
    """The `count` shortest tours leave (strong elitist exchange)."""

    def select_migrants(
        self,
        population: Sequence[Individual],
        cities: Sequence[City],
        count: int,
        rng: random.Random,
    ) -> List[Individual]:
        ranked = sorted(population, key=lambda ind: get_distance(ind, cities))
        return [ind.copy() for ind in ranked[:count]]


class RandomMigrants(MigrantSelectionStrategy):
    """`count` random individuals leave (weaker pressure, more diversity)."""

    def select_migrants(
        self,
        population: Sequence[Individual],
        cities: Sequence[City],
        count: int,
        rng: random.Random,
    ) -> List[Individual]:
        return [ind.copy() for ind in rng.sample(list(population), count)]


# ----------------------------- replacement --------------------------------


def _limit_arrivals(
    population: Sequence[Individual], incoming: Sequence[Individual], cities: Sequence[City]
) -> List[Individual]:
    """Keep at most half of an island's size, preferring the best arrivals.

    With dense topologies (e.g. fully connected) an island can receive
    far more migrants than it has slots; without a cap, immigrants
    would wipe out the whole local population and destroy the very
    diversity the island model exists to protect.
    """
    limit = max(1, len(population) // 2)
    return sorted(incoming, key=lambda ind: get_distance(ind, cities))[:limit]


class ReplaceWorst(ReplacementStrategy):
    """Migrants overwrite the worst (longest-tour) individuals."""

    def replace(
        self,
        population: Sequence[Individual],
        incoming: Sequence[Individual],
        cities: Sequence[City],
        rng: random.Random,
    ) -> List[Individual]:
        incoming = _limit_arrivals(population, incoming, cities)
        ranked = sorted(population, key=lambda ind: get_distance(ind, cities))
        survivors = ranked[: len(population) - len(incoming)]
        return survivors + incoming


class ReplaceRandom(ReplacementStrategy):
    """Migrants overwrite randomly chosen individuals."""

    def replace(
        self,
        population: Sequence[Individual],
        incoming: Sequence[Individual],
        cities: Sequence[City],
        rng: random.Random,
    ) -> List[Individual]:
        incoming = _limit_arrivals(population, incoming, cities)
        keep_indices = set(rng.sample(range(len(population)), len(population) - len(incoming)))
        survivors = [ind for i, ind in enumerate(population) if i in keep_indices]
        return survivors + incoming
