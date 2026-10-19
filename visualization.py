"""Visualization of routes and fitness evolution (matplotlib, headless-safe)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt

from typing import Optional, Sequence

from .genetic_algorithm import GAHistory
from .models import City


def plot_route(cities: Sequence[City], order: Sequence[int], title: str, save_path: str) -> None:
    xs = [cities[i].x for i in order] + [cities[order[0]].x]
    ys = [cities[i].y for i in order] + [cities[order[0]].y]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(xs, ys, "-o", markersize=3 if len(cities) > 150 else 5, linewidth=1)
    ax.plot(xs[0], ys[0], "gs", markersize=10, label="start")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_fitness_history(history: GAHistory, title: str, save_path: str) -> None:
    gens = range(len(history.best_distance))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, history.best_distance, label="best distance")
    ax.plot(gens, history.avg_distance, label="average distance", linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("generation")
    ax.set_ylabel("tour distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_island_history(
    island_histories: Sequence[GAHistory],
    global_history: GAHistory,
    title: str,
    save_path: str,
    migration_interval: Optional[int] = None,
) -> None:
    """Best distance of every island, the global best/average, and migration moments."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for idx, hist in enumerate(island_histories):
        ax.plot(range(len(hist.best_distance)), hist.best_distance,
                linewidth=1, alpha=0.7, label=f"island {idx + 1} best")
    gens = range(len(global_history.best_distance))
    ax.plot(gens, global_history.best_distance, color="black", linewidth=2, label="global best")
    ax.plot(gens, global_history.avg_distance, color="black", linestyle="--", linewidth=1.2,
            label="global average")
    if migration_interval:
        for g in range(migration_interval, len(global_history.best_distance), migration_interval):
            ax.axvline(g, color="grey", linestyle=":", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("generation")
    ax.set_ylabel("tour distance")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
