"""Island-model genetic algorithm.

Several sub-populations ("islands") evolve independently - each island
is a full `GeneticAlgorithm` with its own selection / crossover /
mutation / elitism strategies and its own random stream. Every
`migration_interval` generations a few individuals travel between
islands (migration). Isolation keeps genetic diversity high (islands
explore different regions of the search space and resist premature
convergence), while migration spreads good building blocks between
islands.

`IslandGeneticAlgorithm` depends only on abstractions:
  * `MigrationTopology`, `MigrantSelectionStrategy`, `ReplacementStrategy`
    (migration policy, injected via the constructor);
  * `GeneticAlgorithm.initial_population()` / `.evolve()` (the island's
    single-generation step).
Islands may even be heterogeneous (e.g. one island with tournament
selection, another with roulette) - the driver does not care.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .fitness import evaluate_population, get_distance
from .genetic_algorithm import GAHistory, GeneticAlgorithm
from .interfaces import (
    CrossoverStrategy,
    ElitismStrategy,
    MigrantSelectionStrategy,
    MigrationTopology,
    MutationStrategy,
    ReplacementStrategy,
    SelectionStrategy,
)
from .models import City, Individual


@dataclass
class IslandResult:
    best_individual: Individual
    best_distance: float
    history: GAHistory                  # best / average over ALL islands
    island_histories: List[GAHistory]   # best / average per island
    generations_run: int
    converged_at: Optional[int]
    migrations: int                     # number of migration events performed
    migration_interval: int


class IslandGeneticAlgorithm:
    def __init__(
        self,
        islands: Sequence[GeneticAlgorithm],
        topology: MigrationTopology,
        migrant_selection: MigrantSelectionStrategy,
        replacement: ReplacementStrategy,
        migration_interval: int = 20,
        migration_size: int = 2,
        num_generations: int = 400,
        stagnation_limit: Optional[int] = 60,
        seed: Optional[int] = None,
    ):
        """
        islands: ready-made GA instances, one per island. Each island's
            own `num_generations` / `stagnation_limit` are ignored - the
            island driver controls the global run length and stopping.
        migration_interval: migrate every this many generations.
        migration_size: individuals each island sends per migration
            (to *each* target island of the topology).
        stagnation_limit: stop when the best tour over all islands has
            not improved for this many generations (None = never).
        """
        if len(islands) < 2:
            raise ValueError("an island model needs at least 2 islands")
        if migration_interval < 1:
            raise ValueError("migration_interval must be >= 1")
        smallest = min(isl.population_size for isl in islands)
        if not 1 <= migration_size < smallest:
            raise ValueError("migration_size must be >= 1 and smaller than the smallest island population")
        base_cities = islands[0].cities
        if any(isl.cities != base_cities for isl in islands[1:]):
            raise ValueError("all islands must solve the same set of cities")

        self.islands = list(islands)
        self.cities = base_cities
        self.topology = topology
        self.migrant_selection = migrant_selection
        self.replacement = replacement
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        self.num_generations = num_generations
        self.stagnation_limit = stagnation_limit
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def _migrate(self, populations: List[List[Individual]]) -> List[List[Individual]]:
        """Synchronous migration: everyone's emigrants are chosen from the
        pre-migration populations, then all arrivals are applied, so the
        outcome does not depend on island processing order."""
        n = len(populations)
        outgoing = [
            self.migrant_selection.select_migrants(pop, self.cities, self.migration_size, self.rng)
            for pop in populations
        ]

        incoming: List[List[Individual]] = [[] for _ in range(n)]
        for source, migrants in enumerate(outgoing):
            for target in self.topology.targets(source, n, self.rng):
                incoming[target].extend(m.copy() for m in migrants)

        result: List[List[Individual]] = []
        for pop, arrivals in zip(populations, incoming):
            if arrivals:
                evaluate_population(arrivals, self.cities)
                pop = self.replacement.replace(pop, arrivals, self.cities, self.rng)
            result.append(list(pop))
        return result

    def _best_of(self, populations: Sequence[Sequence[Individual]]) -> Individual:
        return min(
            (ind for pop in populations for ind in pop),
            key=lambda ind: get_distance(ind, self.cities),
        )

    def _record(
        self,
        populations: Sequence[Sequence[Individual]],
        island_histories: List[GAHistory],
        global_history: GAHistory,
    ) -> None:
        for pop, hist in zip(populations, island_histories):
            hist.record(pop, self.cities)
        global_history.record([ind for pop in populations for ind in pop], self.cities)

    # ------------------------------------------------------------------
    def run(self) -> IslandResult:
        populations = [isl.initial_population() for isl in self.islands]
        island_histories = [GAHistory() for _ in self.islands]
        global_history = GAHistory()
        self._record(populations, island_histories, global_history)

        best_ever = self._best_of(populations).copy()
        best_ever_distance = get_distance(best_ever, self.cities)
        stagnant = 0
        converged_at: Optional[int] = None
        migrations = 0
        gen = 0

        for gen in range(1, self.num_generations + 1):
            populations = [isl.evolve(pop) for isl, pop in zip(self.islands, populations)]

            if gen % self.migration_interval == 0:
                populations = self._migrate(populations)
                migrations += 1

            self._record(populations, island_histories, global_history)

            gen_best = self._best_of(populations)
            gen_best_distance = get_distance(gen_best, self.cities)
            if gen_best_distance < best_ever_distance - 1e-9:
                best_ever = gen_best.copy()
                best_ever_distance = gen_best_distance
                stagnant = 0
            else:
                stagnant += 1

            if self.stagnation_limit is not None and stagnant >= self.stagnation_limit:
                converged_at = gen
                break

        return IslandResult(
            best_individual=best_ever,
            best_distance=best_ever_distance,
            history=global_history,
            island_histories=island_histories,
            generations_run=gen,
            converged_at=converged_at,
            migrations=migrations,
            migration_interval=self.migration_interval,
        )


def build_islands(
    cities: Sequence[City],
    num_islands: int,
    island_population: int,
    selection_factory: Callable[[], SelectionStrategy],
    crossover_factory: Callable[[], CrossoverStrategy],
    mutation_factory: Callable[[], MutationStrategy],
    elitism_factory: Callable[[], ElitismStrategy],
    mutation_rate: float = 0.02,
    crossover_rate: float = 0.9,
    seed: Optional[int] = None,
) -> List[GeneticAlgorithm]:
    """Create `num_islands` independent GA instances.

    Strategies are passed as *factories* so every island owns its own
    strategy objects. Each island gets a distinct random stream derived
    from `seed`, so islands explore different regions of the search
    space but the whole run stays reproducible.
    """
    base = 0 if seed is None else seed
    return [
        GeneticAlgorithm(
            cities=cities,
            selection_strategy=selection_factory(),
            crossover_strategy=crossover_factory(),
            mutation_strategy=mutation_factory(),
            elitism_strategy=elitism_factory(),
            population_size=island_population,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            stagnation_limit=None,
            seed=None if seed is None else base + 7919 * (i + 1),
        )
        for i in range(num_islands)
    ]
