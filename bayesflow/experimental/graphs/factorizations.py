import itertools
from collections import defaultdict
from dataclasses import dataclass

import sympy as sp

from .expanded_graph import ExpandedGraph
from .inverted_graph import InvertedGraph
from .simulation_graph import SimulationNode


@dataclass
class Factorization:
    order: tuple[SimulationNode, ...]
    graph: InvertedGraph


def select_factorization(factorizations: list[Factorization]):
    """
    Selects an inverse factorization based on the minimum expected number of sequentially inferred groups,
    using the number of inference networks and number of summary networks as potential tie breakers.
    """
    simulation_graph = factorizations[0].graph.simulation_graph
    meta_fn = simulation_graph.meta_fn
    draws = [meta_fn() for _ in range(1000)] if meta_fn else [{}]
    average_meta = {sp.Symbol(key): sum(d[key] for d in draws) / 1000 for key in draws[0]}

    def cost(factorization):
        total_nodes = set(itertools.chain.from_iterable(factorization.order))
        amortizable_nodes = set(factorization.graph.amortizable_nodes())
        non_amortizable_nodes = total_nodes - amortizable_nodes
        reps = [simulation_graph.nodes[node]["reps"] for node in non_amortizable_nodes]

        sequentially = sp.Add(*[sp.Symbol(r) if isinstance(r, str) else sp.Integer(r) for r in reps])
        inference_networks = len(factorization.graph.inference_variable_shapes())
        summary_networks = len(factorization.graph.summary_network_input_shapes())

        return (round(float(sequentially.subs(average_meta.items()))), inference_networks, summary_networks)

    return min(factorizations, key=cost)


def enumerate_factorizations(expanded_graph: ExpandedGraph):
    """
    Returns a list of all possible inverse factorizations.
    """
    node_mapping = node_name_mapping(expanded_graph)

    latent_nodes = [
        group for group, nodes in node_mapping.items() if any(expanded_graph.out_degree(n) != 0 for n in nodes)
    ]

    factorizations = []

    for permutation in itertools.permutations(latent_nodes):
        expanded_nodes = to_expanded_nodes(list(permutation), node_mapping)
        inverted_graph = expanded_graph.invert(node_ordering=expanded_nodes)

        factorization = Factorization(permutation, inverted_graph)
        factorizations.append(factorization)

    return factorizations


def node_name_mapping(expanded_graph: ExpandedGraph):
    """
    Creates a dictionary with node names of the simulation graph as keys, and a list of
    corresponding node names in the expanded graph as values.
    """
    original_node_names = expanded_graph.original_node_names()
    name_mapping = defaultdict(list)

    for expanded, original in original_node_names.items():
        original = tuple(original) if isinstance(original, list) else (original,)
        name_mapping[original].append(expanded)

    name_mapping = dict(name_mapping)

    return name_mapping


def to_expanded_nodes(node_list: list[SimulationNode], node_mapping: dict):
    """
    Takes a list of nodes in the simulation graph and converts it into a list of nodes
    in the expanded graph. For this, simulation nodes are replaced with all splits
    of expanded nodes.
    """
    expanded_nodes = []
    for node in node_list:
        expanded_nodes.extend(node_mapping[node])

    return expanded_nodes
