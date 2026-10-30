from copy import deepcopy
from typing import TYPE_CHECKING, NamedTuple, TypeAlias

import networkx as nx

from .inverted_graph import InvertedGraph
from .factorizations import enumerate_factorizations, select_factorization

if TYPE_CHECKING:
    from .simulation_graph import SimulationGraph

SimulationNode: TypeAlias = str


class ExpandedNode(NamedTuple):
    """
    A node of an ExpandedGraph or InvertedGraph.

    Attributes
    ----------
    original : str
        The name of the node in the SimulationGraph.
    index : tuple[tuple[str, int], ...]
        Which copy the node is: one ``(node, copy)`` pair for every split node
        it descends from.
    """

    original: str
    index: tuple[tuple[str, int], ...] = ()


class ExpandedGraph(nx.DiGraph):
    """
    A SimulationGraph in which every repeated latent node with
    parents is shown twice.

    The nodes are :py:class:`ExpandedNode` objects, which hold the name of the
    original node and which copy they are. The graph is created by
    :py:meth:`SimulationGraph.expand` and used to find the factorization of the
    posterior.

    Parameters
    ----------
    incoming_graph_data : optional
        Data to initialize the graph with, passed to :py:class:`networkx.DiGraph`.
    simulation_graph : SimulationGraph, optional
        A copy of the graph that was expanded.
    **kwargs
        Additional keyword arguments passed to :py:class:`networkx.DiGraph`.
    """

    def __init__(self, incoming_graph_data=None, *, simulation_graph: "SimulationGraph | None" = None, **kwargs):
        super().__init__(incoming_graph_data, **kwargs)
        self.simulation_graph = deepcopy(simulation_graph)

    def invert(self, ordering: list[SimulationNode] | None = None):
        """
        Inverts the graph into the factorization of the posterior used by
        the MultilevelEstimator.

        The latent nodes are inferred one after the other, starting from the
        data. Every instance of a latent node is conditioned on the fewest
        nodes that are already known that makes it independent of the remaining
        known nodes.

        Parameters
        ----------
        ordering : list[str], optional
            The order in which latent nodes are inferred. Must contain every
            latent node. By default, every order is tried, and the
            one with the fewest expected instances that have to be inferred
            sequentially is chosen.

        Returns
        -------
        InvertedGraph
            The inverted graph, to be passed to the MultilevelEstimator.
        """
        if ordering is None:
            return select_factorization(enumerate_factorizations(self))

        inverted = nx.DiGraph()
        for node in self:
            if self.out_degree(node) == 0:
                inverted.add_node(node)

        for variable in ordering:
            for node in instances(self, variable):
                conditions = minimal_conditions(self, node, known=set(inverted))

                inverted.add_node(node)
                for condition in conditions:
                    inverted.add_edge(condition, node)

        return InvertedGraph(inverted, expanded_graph=self, ordering=ordering)


def minimal_conditions(graph: nx.DiGraph, node: ExpandedNode, known: set[ExpandedNode]) -> set[ExpandedNode]:
    # drops known nodes one by one, as long as the node stays independent of all dropped nodes
    conditions = set(known)

    for candidate in sorted(known):
        fewer = conditions - {candidate}
        if nx.is_d_separator(graph, {node}, known - fewer, fewer):
            conditions = fewer

    return conditions


def instances(graph, variable: SimulationNode) -> list[ExpandedNode]:
    return [node for node in graph if node.original == variable]
