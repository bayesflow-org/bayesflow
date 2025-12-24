from typing import TypeAlias

import networkx as nx

from .simulation_graph import SimulationGraph
from .utils import has_open_path, merge_root_nodes

from copy import deepcopy

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


class ExpandedGraph(nx.DiGraph):
    """
    Directed graph with a similar structure as the graph defined in `SimulationGraph`,
    but in which interior nodes are split into two subgraphs.

    This is necessary to determine if variables from a node can be estimated
    group-wise (enabling amortization over groups).
    """

    def __init__(self, graph_data=None, *, simulation_graph: SimulationGraph):
        super().__init__(graph_data)  # optionally initializing with existing data
        self.simulation_graph = deepcopy(simulation_graph)

    def invert(self, merge_roots: bool = True):
        """
        Inverts a graph by following the algorithm described by [1], but sorting
        latent nodes by outer nodes first.

        [1] Stuhlmüller, A., Taylor, J., & Goodman, N. D. (2013). Learning stochastic inverses.
        In Advances in Neural Information Processing Systems (pp. 3048–3056).
        """
        from .inverted_graph import InvertedGraph

        graph = deepcopy(self)
        if merge_roots:
            graph = merge_root_nodes(graph)

        undirected = graph.to_undirected()
        leaf_nodes = [node for node in graph.nodes() if graph.out_degree(node) == 0]

        # Sort nodes by outer nodes first (instead of last as in Stuhlmüller2013).
        # We assume that this ordering preserves amortization over exchangeable nodes in most cases.
        latent_nodes = [
            node for node in list(nx.lexicographical_topological_sort(graph)) if graph.out_degree(node) != 0
        ]

        inverse = InvertedGraph(expanded_graph=self)
        inverse.add_nodes_from(leaf_nodes)

        for x_j in latent_nodes:
            inverse.add_node(x_j)

            # Iterate over all already added nodes in inverse (shortest distance from x_j first)
            # and check if the path between that node and x_j is blocked.
            # If it is open, draw an edge from that node to x_j.
            other_nodes = [node for node in inverse.nodes() if node != x_j]
            lengths = [nx.shortest_path_length(undirected, x_j, node) for node in other_nodes]
            sorted_nodes = [node for _, node in sorted(zip(lengths, other_nodes))]

            for node in sorted_nodes:
                if has_open_path(graph, x_j, node, other_nodes):
                    inverse.add_edge(node, x_j)

        return inverse
