"""Genetic algorithm for the Travelling Salesman Problem (TSP).

Package built around SOLID principles:
- Single Responsibility: each module has one job (models, fitness,
  strategies, orchestration, visualization, generation, experiments).
- Open/Closed: new selection/crossover/mutation/elitism strategies can
  be added by implementing the interfaces in `interfaces.py`, with no
  change needed to `genetic_algorithm.py`.
- Liskov Substitution: any concrete strategy can replace another of
  the same interface without breaking the algorithm.
- Interface Segregation: separate, narrow interfaces per strategy kind
  instead of one big "GA strategy" interface.
- Dependency Inversion: `GeneticAlgorithm` depends only on the
  abstract strategy interfaces, injected through its constructor.
"""

__version__ = "1.0.0"
