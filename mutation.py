"""Mutation operators for permutation chromosomes.

A simple GA mutates single genes independently (e.g. bit-flip). On a
permutation chromosome that would break validity, so TSP mutation
must swap/rearrange positions instead - another key encoding
difference this lab asks students to articulate.
"""

from __future__ import annotations

import random

from .interfaces import MutationStrategy
from .models import Individual


class SwapMutation(MutationStrategy):
    """With probability `mutation_rate`, swap two random positions."""

    def mutate(self, individual: Individual, mutation_rate: float, rng: random.Random) -> Individual:
        if rng.random() < mutation_rate:
            chrom = individual.chromosome
            i, j = rng.sample(range(len(chrom)), 2)
            chrom[i], chrom[j] = chrom[j], chrom[i]
            individual.chromosome = chrom
        return individual


class InversionMutation(MutationStrategy):
    """With probability `mutation_rate`, reverse a random sub-segment.

    Generally more effective than swap mutation for TSP because it
    preserves more of the existing edge structure (fewer edges broken
    per mutation).
    """

    def mutate(self, individual: Individual, mutation_rate: float, rng: random.Random) -> Individual:
        if rng.random() < mutation_rate:
            chrom = individual.chromosome
            i, j = sorted(rng.sample(range(len(chrom)), 2))
            chrom[i : j + 1] = list(reversed(chrom[i : j + 1]))
            individual.chromosome = chrom
        return individual
