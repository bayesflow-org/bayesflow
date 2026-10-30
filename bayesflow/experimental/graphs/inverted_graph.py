import importlib
import sys
from typing import TYPE_CHECKING, TypeAlias

import networkx as nx

from bayesflow.utils.serialization import serializable

if TYPE_CHECKING:
    from .expanded_graph import ExpandedGraph, ExpandedNode

SimulationNode: TypeAlias = str


@serializable("bayesflow.experimental")
class InvertedGraph(nx.DiGraph):
    """
    The factorization of the posterior that the MultilevelEstimator learns.

    The nodes are the :py:class:`ExpandedNode` objects of the expanded graph,
    and every edge points from a node to an instance it is conditioned on when
    that instance is inferred. The graph is created by
    :py:meth:`ExpandedGraph.invert`. Use :py:meth:`networks` to see which
    nodes each inference network infers, and :py:meth:`conditions` to see what it
    conditions on.

    Parameters
    ----------
    incoming_graph_data : optional
        Data to initialize the graph with, passed to :py:class:`networkx.DiGraph`.
    expanded_graph : ExpandedGraph, optional
        The expanded graph that was inverted.
    ordering : list[str], optional
        The order in which the latent nodes are inferred.
    **kwargs
        Additional keyword arguments passed to :py:class:`networkx.DiGraph`.
    """

    def __init__(
        self,
        incoming_graph_data=None,
        *,
        expanded_graph: "ExpandedGraph | None" = None,
        ordering: list[SimulationNode] | None = None,
        **kwargs,
    ):
        super().__init__(incoming_graph_data, **kwargs)
        self.expanded_graph = expanded_graph
        self.ordering = ordering

    def networks(self) -> list[list[SimulationNode]]:
        """
        Lists the nodes that each inference network infers, in the order they
        are inferred.

        Every node is inferred by its own network, except for root nodes of the
        simulation graph, which are inferred jointly by a single network.

        Returns
        -------
        list[list[str]]
            The nodes of every network. The MultilevelEstimator expects one
            inference network for each entry, in the same order.
        """
        simulation_graph = self.expanded_graph.simulation_graph

        networks = []

        for variable in self.ordering:
            is_root = simulation_graph.in_degree(variable) == 0
            follows_root = networks and simulation_graph.in_degree(networks[-1][-1]) == 0

            if is_root and follows_root:
                networks[-1].append(variable)
            else:
                networks.append([variable])

        return networks

    def conditions(self, target_nodes: list[SimulationNode]) -> list[SimulationNode]:
        """
        Lists the nodes that an inference network conditions on.

        These are the data nodes and the nodes of earlier networks that the
        posterior of the network's nodes depends on. Nodes that are independent
        of the network's nodes, given the others, are left out.

        Parameters
        ----------
        target_nodes : list[str]
            The nodes of one network, as returned by :py:meth:`networks`.

        Returns
        -------
        list[str]
            The nodes the network conditions on, in topological order of the
            simulation graph.
        """
        parents = set()

        for variable in target_nodes:
            for node in instances(self, variable):
                for parent in self.predecessors(node):
                    parents.add(parent.original)

        simulation_graph = self.expanded_graph.simulation_graph

        result = []
        for variable in nx.lexicographical_topological_sort(simulation_graph):
            if variable in parents and variable not in target_nodes:
                result.append(variable)

        return result

    def is_amortizable(self, variable: SimulationNode) -> bool:
        """
        Checks whether all instances of a node can be inferred at once.

        A node is amortizable if none of its instances is conditioned on another
        instance of the same node. Otherwise, the MultilevelEstimator infers
        the instances one after another.

        Parameters
        ----------
        variable : str
            The name of a node in the simulation graph.

        Returns
        -------
        bool
            True if all instances can be inferred at once.
        """
        # a variable is amortizable if none of its instances is conditioned on another one
        nodes = instances(self, variable)

        for node in nodes:
            for other in nodes:
                if self.has_edge(node, other):
                    return False

        return True

    def get_config(self):
        simulation_graph = self.expanded_graph.simulation_graph

        reps = {}

        for node in simulation_graph:
            node_reps = simulation_graph.nodes[node]["reps"]

            if callable(node_reps):
                node_reps = {"function": function_reference(node_reps)}

            reps[node] = node_reps

        return {"reps": reps, "edges": list(simulation_graph.edges), "ordering": self.ordering}

    @classmethod
    def from_config(cls, config):
        from .simulation_graph import SimulationGraph

        simulation_graph = SimulationGraph()

        for node, node_reps in config["reps"].items():
            if isinstance(node_reps, dict):
                node_reps = import_function(node_reps["function"])

            simulation_graph.add_node(node, reps=node_reps)

        for parent, child in config["edges"]:
            simulation_graph.add_edge(parent, child)

        return simulation_graph.expand().invert(config["ordering"])


def function_reference(fn) -> str:
    module = sys.modules[fn.__module__]

    if getattr(module, fn.__name__, None) is not fn:
        raise ValueError(f"Cannot serialize {fn}: callable reps must be top-level functions.")

    return f"{fn.__module__}:{fn.__name__}"


def import_function(reference: str):
    module, name = reference.split(":")

    return getattr(importlib.import_module(module), name)


def instances(graph, variable: SimulationNode) -> list["ExpandedNode"]:
    return [node for node in graph if node.original == variable]
