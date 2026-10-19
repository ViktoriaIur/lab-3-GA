"""Experiment harness: repeated runs with statistics, and the full
suite of experiments required by the lab checklist.
"""

from __future__ import annotations

import csv
import os
import statistics
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .city_generators import generate_clustered_cities, generate_line_cities
from .crossover import OrderCrossover
from .elitism import NoElitism, StandardElitism
from .genetic_algorithm import GeneticAlgorithm
from .models import City
from .mutation import InversionMutation
from .selection import RouletteWheelSelection, TournamentSelection
from .visualization import plot_fitness_history, plot_route


@dataclass
class RunStat:
    label: str
    best_distances: List[float]
    runtimes_sec: List[float]
    converged_generations: List[int]

    @property
    def mean(self) -> float:
        return statistics.mean(self.best_distances)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.best_distances) if len(self.best_distances) > 1 else 0.0

    @property
    def best(self) -> float:
        return min(self.best_distances)

    @property
    def mean_runtime(self) -> float:
        return statistics.mean(self.runtimes_sec)


def run_repeated(
    label: str,
    cities: Sequence[City],
    selection_strategy,
    elitism_strategy,
    num_runs: int = 5,
    population_size: int = 150,
    num_generations: int = 400,
    mutation_rate: float = 0.02,
    stagnation_limit: Optional[int] = 60,
    out_dir: Optional[str] = None,
) -> RunStat:
    """Run the same configuration `num_runs` times (efficiency study).

    'Efficiency' here is measured along three axes, as required by the
    lab: (1) solution quality - best tour distance found, mean and
    std-dev across runs; (2) convergence speed - the generation at
    which the stagnation-based stop condition triggered; (3) wall-
    clock runtime per run.
    """
    best_distances: List[float] = []
    runtimes: List[float] = []
    converged_gens: List[int] = []
    best_result = None
    best_result_distance = float("inf")

    for run_idx in range(num_runs):
        ga = GeneticAlgorithm(
            cities=cities,
            selection_strategy=selection_strategy,
            crossover_strategy=OrderCrossover(),
            mutation_strategy=InversionMutation(),
            elitism_strategy=elitism_strategy,
            population_size=population_size,
            num_generations=num_generations,
            mutation_rate=mutation_rate,
            stagnation_limit=stagnation_limit,
            seed=1000 + run_idx,
        )
        t0 = time.perf_counter()
        result = ga.run()
        elapsed = time.perf_counter() - t0

        best_distances.append(result.best_distance)
        runtimes.append(elapsed)
        converged_gens.append(result.converged_at or result.generations_run)

        if result.best_distance < best_result_distance:
            best_result_distance = result.best_distance
            best_result = result

    if out_dir and best_result is not None:
        os.makedirs(out_dir, exist_ok=True)
        safe_label = label.replace(" ", "_").replace("/", "-")
        plot_route(
            cities,
            best_result.best_individual.chromosome,
            f"{label} - best route (dist={best_result.best_distance:.1f})",
            os.path.join(out_dir, f"{safe_label}_route.png"),
        )
        plot_fitness_history(
            best_result.history,
            f"{label} - fitness history",
            os.path.join(out_dir, f"{safe_label}_fitness.png"),
        )

    return RunStat(label, best_distances, runtimes, converged_gens)


def write_stats_csv(stats: List[RunStat], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "mean_distance", "stdev_distance", "best_distance", "mean_runtime_sec", "avg_converged_gen"])
        for s in stats:
            writer.writerow([
                s.label,
                f"{s.mean:.3f}",
                f"{s.stdev:.3f}",
                f"{s.best:.3f}",
                f"{s.mean_runtime:.3f}",
                f"{statistics.mean(s.converged_generations):.1f}",
            ])


def run_full_suite(
    out_dir: str = "results",
    city_counts: Sequence[int] = (100, 500),
    num_runs: int = 5,
    population_size: int = 150,
    num_generations: int = 400,
    quick: bool = False,
) -> List[RunStat]:
    """Run the full experiment matrix required by the lab checklist:

    - test cases: clustered ("AI-proposed") and line layout
    - city counts: 100 and 500
    - selection: roulette, tournament
    - elitism: none, and 2-3 elitism parameter sets
    (5 repeated runs per configuration, as required)

    `quick=True` shrinks population/generations for a fast smoke test
    of the whole pipeline; use `quick=False` (default) for the real
    report-quality results, which will take considerably longer,
    especially for n=500.
    """
    if quick:
        population_size, num_generations, num_runs = 30, 40, 2

    os.makedirs(out_dir, exist_ok=True)
    all_stats: List[RunStat] = []

    test_cases = {
        "ai_clustered": lambda n: generate_clustered_cities(n, seed=42),
        "line": lambda n: generate_line_cities(n),
    }

    selection_variants = {
        "roulette": lambda: RouletteWheelSelection(),
        "tournament": lambda: TournamentSelection(tournament_size=5),
    }

    elitism_variants = {
        "no_elitism": lambda: NoElitism(),
        "elitism_2": lambda: StandardElitism(count=2),
        "elitism_5pct": lambda: StandardElitism(count=max(1, population_size // 20)),
    }

    for n in city_counts:
        for case_name, case_fn in test_cases.items():
            cities = case_fn(n)
            for sel_name, sel_fn in selection_variants.items():
                for elit_name, elit_fn in elitism_variants.items():
                    label = f"n{n}_{case_name}_{sel_name}_{elit_name}"
                    print(f"Running: {label}")
                    stat = run_repeated(
                        label=label,
                        cities=cities,
                        selection_strategy=sel_fn(),
                        elitism_strategy=elit_fn(),
                        num_runs=num_runs,
                        population_size=population_size,
                        num_generations=num_generations,
                        out_dir=os.path.join(out_dir, "plots"),
                    )
                    all_stats.append(stat)

    write_stats_csv(all_stats, os.path.join(out_dir, "summary.csv"))
    return all_stats


def run_island_comparison(
    out_dir: str = "results",
    num_runs: int = 5,
    num_cities: int = 100,
    num_islands: int = 4,
    island_population: int = 40,
    num_generations: int = 400,
    migration_interval: int = 20,
    migration_size: int = 3,
    quick: bool = False,
) -> List[RunStat]:
    """Island model vs. a classic single-population GA, at equal cost.

    Fairness: every configuration uses the same total population
    (`num_islands * island_population`), the same number of generations,
    the same operators (tournament k=5, OX1, inversion, elitism 2) and no
    early stopping, so the final tour lengths are directly comparable.
    """
    from .island_model import IslandGeneticAlgorithm, build_islands
    from .migration import BestMigrants, FullyConnectedTopology, RingTopology, ReplaceWorst

    if quick:
        num_runs, num_cities, island_population, num_generations = 2, 30, 15, 40
        migration_interval, migration_size = 10, 2

    os.makedirs(out_dir, exist_ok=True)
    cities = generate_clustered_cities(num_cities, seed=42)
    total_population = num_islands * island_population

    def run_single(seed: int) -> float:
        ga = GeneticAlgorithm(
            cities=cities,
            selection_strategy=TournamentSelection(tournament_size=5),
            crossover_strategy=OrderCrossover(),
            mutation_strategy=InversionMutation(),
            elitism_strategy=StandardElitism(count=2),
            population_size=total_population,
            num_generations=num_generations,
            stagnation_limit=None,
            seed=seed,
        )
        return ga.run().best_distance

    def run_island(topology_cls, seed: int) -> float:
        islands = build_islands(
            cities, num_islands, island_population,
            selection_factory=lambda: TournamentSelection(tournament_size=5),
            crossover_factory=OrderCrossover,
            mutation_factory=InversionMutation,
            elitism_factory=lambda: StandardElitism(count=2),
            seed=seed,
        )
        model = IslandGeneticAlgorithm(
            islands, topology_cls(), BestMigrants(), ReplaceWorst(),
            migration_interval=migration_interval, migration_size=migration_size,
            num_generations=num_generations, stagnation_limit=None, seed=seed,
        )
        return model.run().best_distance

    variants = {
        f"single_pop{total_population}": run_single,
        f"island{num_islands}_ring": lambda seed: run_island(RingTopology, seed),
        f"island{num_islands}_full": lambda seed: run_island(FullyConnectedTopology, seed),
    }

    stats: List[RunStat] = []
    for name, runner in variants.items():
        label = f"n{num_cities}_{name}"
        print(f"Running: {label}")
        distances, runtimes = [], []
        for run_idx in range(num_runs):
            t0 = time.perf_counter()
            distances.append(runner(1000 + run_idx))
            runtimes.append(time.perf_counter() - t0)
        stats.append(RunStat(label, distances, runtimes, [num_generations] * num_runs))

    write_stats_csv(stats, os.path.join(out_dir, "island_comparison.csv"))
    return stats
