"""Generators for the required test cases.

Two kinds of layouts are required by the assignment:
1. An "AI-proposed" test case - here, a clustered layout (several
   Gaussian blobs), which is a realistic instance an LLM would
   commonly suggest to exercise a TSP solver (mimics city clusters /
   regional hubs rather than uniform noise) and is harder than a
   convex/uniform layout because of the inter-cluster routing choices.
2. A degenerate line layout: all cities on a straight line, spaced
   1 unit apart. Its optimal tour is trivially known (traverse in
   order, distance = 2*(n-1)), which makes it a useful correctness/
   sanity benchmark for the algorithm.
"""

from __future__ import annotations

import random
from typing import List

from .models import City


def generate_random_cities(n: int, seed: int | None = None, width: float = 1000.0, height: float = 1000.0) -> List[City]:
    rng = random.Random(seed)
    return [City(i, rng.uniform(0, width), rng.uniform(0, height)) for i in range(n)]


def generate_clustered_cities(
    n: int,
    seed: int | None = None,
    num_clusters: int = 6,
    area: float = 1000.0,
    cluster_spread: float = 40.0,
) -> List[City]:
    """AI-proposed test case: cities grouped into `num_clusters` blobs."""
    rng = random.Random(seed)
    centers = [(rng.uniform(0, area), rng.uniform(0, area)) for _ in range(num_clusters)]
    cities = []
    for i in range(n):
        cx, cy = centers[i % num_clusters]
        x = min(max(rng.gauss(cx, cluster_spread), 0), area)
        y = min(max(rng.gauss(cy, cluster_spread), 0), area)
        cities.append(City(i, x, y))
    return cities


def generate_line_cities(n: int, step: float = 1.0) -> List[City]:
    """Degenerate test case: all cities on a line, `step` apart."""
    return [City(i, i * step, 0.0) for i in range(n)]
