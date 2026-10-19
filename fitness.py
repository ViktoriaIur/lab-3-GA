"""Fitness evaluation for TSP individuals.

Kept as its own module (single responsibility) so that the distance/
fitness definition can change without touching selection, crossover,
mutation, or the GA loop itself.
"""

from __future__ import annotations

from typing import List, Sequence

from .models import City, Individual


def tour_distance(chromosome: Sequence[int], cities: Sequence[City]) -> float:
    """Total closed-loop distance of a tour (returns to the start)."""
    total = 0.0
    n = len(chromosome)
    for i in range(n):
        a = cities[chromosome[i]]
        b = cities[chromosome[(i + 1) % n]]
        total += a.distance_to(b)
    return total


def evaluate(individual: Individual, cities: Sequence[City]) -> float:
    """Compute and cache distance + fitness (1 / distance) on the individual.

    Fitness is defined as the inverse of distance so that *higher is
    better*, which lets selection strategies (roulette, tournament)
    stay distance-metric-agnostic.
    """
    distance = tour_distance(individual.chromosome, cities)
    individual._distance = distance
    individual._fitness = 1.0 / distance if distance > 0 else float("inf")
    return individual._fitness


def get_distance(individual: Individual, cities: Sequence[City]) -> float:
    if individual._distance is None:
        evaluate(individual, cities)
    return individual._distance  # type: ignore[return-value]


def get_fitness(individual: Individual, cities: Sequence[City]) -> float:
    if individual._fitness is None:
        evaluate(individual, cities)
    return individual._fitness  # type: ignore[return-value]


def evaluate_population(population: List[Individual], cities: Sequence[City]) -> None:
    for ind in population:
        evaluate(ind, cities)
