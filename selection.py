"""Concrete selection strategies: roulette-wheel and tournament."""

from __future__ import annotations

import random
from typing import List, Sequence

from .fitness import get_fitness
from .interfaces import SelectionStrategy
from .models import City, Individual


class RouletteWheelSelection(SelectionStrategy):
    """Fitness-proportionate selection.

    Each individual's chance of being picked is proportional to its
    fitness (1 / distance). Susceptible to premature convergence when
    one individual dominates early.
    """

    def select(
        self,
        population: Sequence[Individual],
        cities: Sequence[City],
        num_parents: int,
        rng: random.Random,
    ) -> List[Individual]:
        weights = [get_fitness(ind, cities) for ind in population]
        return list(rng.choices(population, weights=weights, k=num_parents))


class TournamentSelection(SelectionStrategy):
    """Pick the best of `tournament_size` randomly sampled individuals.

    Selection pressure is tunable via `tournament_size`: larger values
    push convergence faster but reduce diversity.
    """

    def __init__(self, tournament_size: int = 3):
        if tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
        self.tournament_size = tournament_size

    def select(
        self,
        population: Sequence[Individual],
        cities: Sequence[City],
        num_parents: int,
        rng: random.Random,
    ) -> List[Individual]:
        parents: List[Individual] = []
        n = len(population)
        k = min(self.tournament_size, n)
        for _ in range(num_parents):
            contenders = rng.sample(range(n), k)
            best = max(contenders, key=lambda i: get_fitness(population[i], cities))
            parents.append(population[best])
        return parents
