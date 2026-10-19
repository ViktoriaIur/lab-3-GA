"""Order-preserving crossover operators for permutation chromosomes."""

from __future__ import annotations

import random
from typing import List

from .interfaces import CrossoverStrategy
from .models import Individual


class OrderCrossover(CrossoverStrategy):
    """OX1 - Order Crossover.

    A contiguous slice is copied verbatim from parent1; the remaining
    positions are filled with parent2's genes in their original
    relative order, skipping genes already present. Guarantees a
    valid permutation child, unlike single/two-point crossover used
    in a simple GA on bit strings.
    """

    def crossover(self, parent1: Individual, parent2: Individual, rng: random.Random) -> Individual:
        size = len(parent1)
        a, b = sorted(rng.sample(range(size), 2))
        child: List[int | None] = [None] * size
        child[a : b + 1] = parent1.chromosome[a : b + 1]
        taken = set(child[a : b + 1])

        fill_positions = [i for i in range(size) if child[i] is None]
        fill_values = [g for g in parent2.chromosome if g not in taken]
        for pos, val in zip(fill_positions, fill_values):
            child[pos] = val

        return Individual([g for g in child])  # type: ignore[misc]


class PartiallyMappedCrossover(CrossoverStrategy):
    """PMX - Partially Mapped Crossover.

    Alternative order-based operator: swaps a slice between parents
    and repairs duplicates via a position mapping. Included to allow
    experimenting with more than one order-based crossover.
    """

    def crossover(self, parent1: Individual, parent2: Individual, rng: random.Random) -> Individual:
        size = len(parent1)
        a, b = sorted(rng.sample(range(size), 2))
        p1, p2 = parent1.chromosome, parent2.chromosome

        child: List[int | None] = [None] * size
        child[a : b + 1] = p1[a : b + 1]
        segment = set(child[a : b + 1])

        # Reverse mapping p2 -> p1 for the copied segment, used to
        # resolve collisions when a p2 gene is already in the segment.
        pos_in_p1 = {gene: idx for idx, gene in enumerate(p1)}

        for i in list(range(0, a)) + list(range(b + 1, size)):
            candidate = p2[i]
            while candidate in segment:
                candidate = p2[pos_in_p1[candidate]]
            child[i] = candidate

        return Individual(child)  # type: ignore[arg-type]
