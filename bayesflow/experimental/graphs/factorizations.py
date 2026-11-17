import itertools
from typing import TYPE_CHECKING

import numpy as np

from .inverted_graph import InvertedGraph

if TYPE_CHECKING:
    from .expanded_graph import ExpandedGraph


def enumerate_factorizations(expanded_graph: "ExpandedGraph") -> list[InvertedGraph]:
    simulation_graph = expanded_graph.simulation_graph

    latent = []
    for variable in sorted(simulation_graph):
        if simulation_graph.out_degree(variable) > 0:
            latent.append(variable)

    factorizations = []
    for ordering in itertools.permutations(latent):
        factorizations.append(expanded_graph.invert(list(ordering)))

    return factorizations


def select_factorization(factorizations: list[InvertedGraph]) -> InvertedGraph:
    simulation_graph = factorizations[0].expanded_graph.simulation_graph
    mean_reps = mean_repetitions(simulation_graph)

    # the cost is the expected number of instances that are inferred sequentially
    costs = []
    for factorization in factorizations:
        cost = 0

        for variable in simulation_graph:
            if not factorization.is_amortizable(variable):
                cost += mean_reps[variable]

        costs.append(cost)

    return factorizations[np.argmin(costs)]


def mean_repetitions(simulation_graph, num_samples: int = 1000) -> dict[str, float]:
    mean_meta = {}
    if simulation_graph.meta_fn:
        meta_samples = []
        for _ in range(num_samples):
            meta_samples.append(simulation_graph.meta_fn())

        for key in meta_samples[0]:
            mean_meta[key] = np.mean([sample[key] for sample in meta_samples])

    mean_reps = {}
    for variable in simulation_graph:
        if simulation_graph.out_degree(variable) == 0:
            continue

        reps = simulation_graph.nodes[variable].get("reps", 1)

        if callable(reps):
            mean_reps[variable] = np.mean(reps(size=num_samples))
        elif isinstance(reps, str):
            mean_reps[variable] = mean_meta[reps]
        else:
            mean_reps[variable] = reps

    return mean_reps
