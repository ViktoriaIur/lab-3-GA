"""Genetic algorithm orchestrator for the TSP.

`GeneticAlgorithm` knows nothing about *how* selection, crossover,
mutation, or elitism are implemented - only that they satisfy the
interfaces in `interfaces.py`. This is Dependency Inversion in
practice: swapping roulette for tournament selection, or plugging in
elitism vs. no elitism, never requires editing this file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .fitness import evaluate_population, get_distance, get_fitness
from .interfaces import CrossoverStrategy, ElitismStrategy, MutationStrategy, SelectionStrategy
from .models import City, Individual
from .elitism import NoElitism


@dataclass
class GAHistory:
    best_distance: List[float] = field(default_factory=list)
    avg_distance: List[float] = field(default_factory=list)

    def record(self, population: Sequence[Individual], cities: Sequence[City]) -> None:
        distances = [get_distance(ind, cities) for ind in population]
        self.best_distance.append(min(distances))
        self.avg_distance.append(sum(distances) / len(distances))


@dataclass
class GAResult:
    best_individual: Individual
    best_distance: float
    history: GAHistory
    generations_run: int
    converged_at: Optional[int]  # generation index the stop condition triggered, if any


class GeneticAlgorithm:
    def __init__(
        self,
        cities: Sequence[City],
        selection_strategy: SelectionStrategy,
        crossover_strategy: CrossoverStrategy,
        mutation_strategy: MutationStrategy,
        elitism_strategy: Optional[ElitismStrategy] = None,
        population_size: int = 100,
        num_generations: int = 300,
        mutation_rate: float = 0.02,
        crossover_rate: float = 0.9,
        stagnation_limit: Optional[int] = 50,
        seed: Optional[int] = None,
    ):
        """
        stagnation_limit: stop early if the best distance does not
        improve (by more than a tiny epsilon) for this many
        consecutive generations. None disables early stopping and the
        GA always runs `num_generations`. This is the algorithm's
        stopping condition, justified because a GA on TSP rarely
        improves monotonically forever - once the best tour plateaus,
        further generations mostly burn compute for negligible gain.
        """
        if population_size < 4:
            raise ValueError("population_size must be >= 4")
        self.cities = list(cities)
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.elitism_strategy = elitism_strategy or NoElitism()
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.stagnation_limit = stagnation_limit
        self.rng = random.Random(seed)

    def _random_individual(self) -> Individual:
        order = list(range(len(self.cities)))
        self.rng.shuffle(order)
        return Individual(order)

    def _init_population(self) -> List[Individual]:
        return [self._random_individual() for _ in range(self.population_size)]

    def initial_population(self) -> List[Individual]:
        """Create and evaluate a random starting population.

        Public so that other drivers (e.g. the island model) can reuse
        this GA as a self-contained "island" without duplicating logic.
        """
        population = self._init_population()
        evaluate_population(population, self.cities)
        return population

    def evolve(self, population: List[Individual]) -> List[Individual]:
        """Produce the next generation from `population` (one GA step).

        Applies elitism, selection, crossover and mutation, and returns
        a new, fully evaluated population of `population_size`
        individuals. The input list is not modified.
        """
        elite = self.elitism_strategy.select_elite(population, self.cities)

        offspring_needed = self.population_size - len(elite)
        children: List[Individual] = []
        while len(children) < offspring_needed:
            p1, p2 = self.selection_strategy.select(population, self.cities, 2, self.rng)
            if self.rng.random() < self.crossover_rate:
                child = self.crossover_strategy.crossover(p1, p2, self.rng)
            else:
                child = p1.copy()
            child = self.mutation_strategy.mutate(child, self.mutation_rate, self.rng)
            children.append(child)

        new_population = elite + children
        evaluate_population(new_population, self.cities)
        return new_population

    def run(self) -> GAResult:
        population = self.initial_population()
        history = GAHistory()
        history.record(population, self.cities)

        best_ever = min(population, key=lambda ind: get_distance(ind, self.cities)).copy()
        best_ever_distance = get_distance(best_ever, self.cities)
        stagnant_generations = 0
        converged_at: Optional[int] = None
        gen = 0

        for gen in range(1, self.num_generations + 1):
            population = self.evolve(population)
            history.record(population, self.cities)

            gen_best = min(population, key=lambda ind: get_distance(ind, self.cities))
            gen_best_distance = get_distance(gen_best, self.cities)
            if gen_best_distance < best_ever_distance - 1e-9:
                best_ever = gen_best.copy()
                best_ever_distance = gen_best_distance
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if self.stagnation_limit is not None and stagnant_generations >= self.stagnation_limit:
                converged_at = gen
                break

        return GAResult(
            best_individual=best_ever,
            best_distance=best_ever_distance,
            history=history,
            generations_run=gen,
            converged_at=converged_at,
        )
