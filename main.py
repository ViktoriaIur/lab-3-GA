"""Command-line entry point.

Usage examples (run from the project root):

    python -m ga_tsp.main demo
    python -m ga_tsp.main demo --cities 200 --selection tournament --elitism 5
    python -m ga_tsp.main experiments --quick
    python -m ga_tsp.main experiments
    python -m ga_tsp.main island --islands 4 --topology ring
    python -m ga_tsp.main island-compare --quick
"""

from __future__ import annotations

import argparse
import os

from .city_generators import generate_clustered_cities, generate_line_cities
from .crossover import OrderCrossover
from .elitism import NoElitism, StandardElitism
from .genetic_algorithm import GeneticAlgorithm
from .island_model import IslandGeneticAlgorithm, build_islands
from .migration import (
    BestMigrants,
    FullyConnectedTopology,
    RandomMigrants,
    RandomTopology,
    ReplaceRandom,
    ReplaceWorst,
    RingTopology,
)
from .mutation import InversionMutation
from .selection import RouletteWheelSelection, TournamentSelection
from .visualization import plot_fitness_history, plot_island_history, plot_route


def cmd_demo(args: argparse.Namespace) -> None:
    cities = (
        generate_line_cities(args.cities)
        if args.layout == "line"
        else generate_clustered_cities(args.cities, seed=args.seed)
    )

    selection = TournamentSelection(tournament_size=5) if args.selection == "tournament" else RouletteWheelSelection()
    elitism = StandardElitism(count=args.elitism) if args.elitism > 0 else NoElitism()

    ga = GeneticAlgorithm(
        cities=cities,
        selection_strategy=selection,
        crossover_strategy=OrderCrossover(),
        mutation_strategy=InversionMutation(),
        elitism_strategy=elitism,
        population_size=args.population,
        num_generations=args.generations,
        mutation_rate=args.mutation_rate,
        stagnation_limit=args.stagnation_limit,
        seed=args.seed,
    )

    print(f"Running GA: {args.cities} cities ({args.layout}), selection={args.selection}, elitism={args.elitism}")
    result = ga.run()
    print(f"Best distance: {result.best_distance:.2f}")
    print(f"Generations run: {result.generations_run} (converged_at={result.converged_at})")

    os.makedirs(args.out, exist_ok=True)
    route_path = os.path.join(args.out, "demo_route.png")
    fitness_path = os.path.join(args.out, "demo_fitness.png")
    plot_route(cities, result.best_individual.chromosome, f"Best route (distance={result.best_distance:.1f})", route_path)
    plot_fitness_history(result.history, "Fitness over generations", fitness_path)
    print(f"Saved: {route_path}")
    print(f"Saved: {fitness_path}")


def cmd_experiments(args: argparse.Namespace) -> None:
    from .experiment import run_full_suite

    run_full_suite(
        out_dir=args.out,
        city_counts=(100, 500) if not args.quick else (20, 50),
        num_runs=args.runs,
        population_size=args.population,
        num_generations=args.generations,
        quick=args.quick,
    )
    print(f"Done. See {args.out}/summary.csv and {args.out}/plots/")


TOPOLOGIES = {"ring": RingTopology, "full": FullyConnectedTopology, "random": RandomTopology}
MIGRANTS = {"best": BestMigrants, "random": RandomMigrants}
REPLACEMENTS = {"worst": ReplaceWorst, "random": ReplaceRandom}


def _make_cities(layout: str, n: int, seed: int):
    return generate_line_cities(n) if layout == "line" else generate_clustered_cities(n, seed=seed)


def cmd_island(args: argparse.Namespace) -> None:
    cities = _make_cities(args.layout, args.cities, args.seed)

    islands = build_islands(
        cities=cities,
        num_islands=args.islands,
        island_population=args.island_population,
        selection_factory=(lambda: TournamentSelection(tournament_size=5))
        if args.selection == "tournament" else RouletteWheelSelection,
        crossover_factory=OrderCrossover,
        mutation_factory=InversionMutation,
        elitism_factory=(lambda: StandardElitism(count=args.elitism)) if args.elitism > 0 else NoElitism,
        mutation_rate=args.mutation_rate,
        seed=args.seed,
    )
    model = IslandGeneticAlgorithm(
        islands=islands,
        topology=TOPOLOGIES[args.topology](),
        migrant_selection=MIGRANTS[args.migrants](),
        replacement=REPLACEMENTS[args.replacement](),
        migration_interval=args.migration_interval,
        migration_size=args.migration_size,
        num_generations=args.generations,
        stagnation_limit=args.stagnation_limit,
        seed=args.seed,
    )

    print(
        f"Island GA: {args.cities} cities ({args.layout}), {args.islands} islands x "
        f"{args.island_population} individuals, topology={args.topology}, "
        f"migrants={args.migrants}/{args.replacement}, every {args.migration_interval} gen, "
        f"size={args.migration_size}"
    )
    result = model.run()
    print(f"Best distance: {result.best_distance:.2f}")
    print(f"Generations run: {result.generations_run} (converged_at={result.converged_at}), "
          f"migrations: {result.migrations}")

    os.makedirs(args.out, exist_ok=True)
    route_path = os.path.join(args.out, "island_route.png")
    fitness_path = os.path.join(args.out, "island_fitness.png")
    plot_route(cities, result.best_individual.chromosome,
               f"Island model - best route (distance={result.best_distance:.1f})", route_path)
    plot_island_history(result.island_histories, result.history,
                        "Island model - fitness over generations", fitness_path,
                        migration_interval=result.migration_interval)
    print(f"Saved: {route_path}")
    print(f"Saved: {fitness_path}")


def cmd_island_compare(args: argparse.Namespace) -> None:
    from .experiment import run_island_comparison

    run_island_comparison(out_dir=args.out, quick=args.quick, num_runs=args.runs,
                          num_cities=args.cities, num_generations=args.generations)
    print(f"Done. See {args.out}/island_comparison.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Genetic algorithm for the TSP")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Run a single GA instance and save plots")
    demo.add_argument("--cities", type=int, default=100)
    demo.add_argument("--layout", choices=["ai_clustered", "line"], default="ai_clustered")
    demo.add_argument("--selection", choices=["roulette", "tournament"], default="tournament")
    demo.add_argument("--elitism", type=int, default=2, help="0 disables elitism")
    demo.add_argument("--population", type=int, default=150)
    demo.add_argument("--generations", type=int, default=400)
    demo.add_argument("--mutation-rate", type=float, default=0.02)
    demo.add_argument("--stagnation-limit", type=int, default=60)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--out", type=str, default="results")
    demo.set_defaults(func=cmd_demo)

    experiments = sub.add_parser("experiments", help="Run the full checklist experiment matrix")
    experiments.add_argument("--quick", action="store_true", help="Small/fast smoke-test run")
    experiments.add_argument("--runs", type=int, default=5)
    experiments.add_argument("--population", type=int, default=150)
    experiments.add_argument("--generations", type=int, default=400)
    experiments.add_argument("--out", type=str, default="results")
    experiments.set_defaults(func=cmd_experiments)

    island = sub.add_parser("island", help="Run the island-model GA and save plots")
    island.add_argument("--cities", type=int, default=100)
    island.add_argument("--layout", choices=["ai_clustered", "line"], default="ai_clustered")
    island.add_argument("--islands", type=int, default=4, help="number of islands (>= 2)")
    island.add_argument("--island-population", type=int, default=40, help="individuals per island")
    island.add_argument("--topology", choices=sorted(TOPOLOGIES), default="ring")
    island.add_argument("--migrants", choices=sorted(MIGRANTS), default="best", help="who leaves an island")
    island.add_argument("--replacement", choices=sorted(REPLACEMENTS), default="worst", help="whom migrants replace")
    island.add_argument("--migration-interval", type=int, default=20, help="migrate every N generations")
    island.add_argument("--migration-size", type=int, default=3, help="migrants per target island")
    island.add_argument("--selection", choices=["roulette", "tournament"], default="tournament")
    island.add_argument("--elitism", type=int, default=2, help="0 disables elitism")
    island.add_argument("--generations", type=int, default=400)
    island.add_argument("--mutation-rate", type=float, default=0.02)
    island.add_argument("--stagnation-limit", type=int, default=60)
    island.add_argument("--seed", type=int, default=42)
    island.add_argument("--out", type=str, default="results")
    island.set_defaults(func=cmd_island)

    compare = sub.add_parser("island-compare", help="Compare the island model with a single-population GA")
    compare.add_argument("--quick", action="store_true", help="Small/fast smoke-test run")
    compare.add_argument("--runs", type=int, default=5)
    compare.add_argument("--cities", type=int, default=100)
    compare.add_argument("--generations", type=int, default=400)
    compare.add_argument("--out", type=str, default="results")
    compare.set_defaults(func=cmd_island_compare)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
