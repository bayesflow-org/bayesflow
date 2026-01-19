from copy import deepcopy
from typing import TypeAlias

import networkx as nx
from networkx.readwrite import json_graph

from bayesflow.utils.serialization import serializable, serialize

from .expanded_graph import ExpandedGraph
from .utils import merge_root_nodes, sort_nodes_topologically

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


@serializable("bayesflow.experimental")
class InvertedGraph(nx.DiGraph):
    """
    Directed graph representing the factorization of the joint posterior of a
    forward model defined by `SimulationGraph`.

    An `InvertedGraph` is derived from an `ExpandedGraph` and encodes the
    dependency structure between variables.
    """

    def __init__(self, *, expanded_graph: ExpandedGraph, graph_data=None, **kwargs):
        super().__init__(**kwargs)  # optionally initializing with existing data

        self.simulation_graph = deepcopy(expanded_graph.simulation_graph)
        self.expanded_graph = deepcopy(expanded_graph)

        if graph_data is not None:
            g = json_graph.node_link_graph(graph_data, directed=True, multigraph=False)

            self.add_nodes_from(g.nodes(data=True))
            self.add_edges_from(g.edges(data=True))

    def network_conditions(self) -> dict[int, list[SimulationNode]]:
        """
        Returns a dictionary where the keys are integer network indices and the values
        are lists of nodes that are required as conditions by that inference network.
        """
        composition = self.network_composition()
        conditions = self.conditions_by_node()

        networks: dict[int, list[SimulationNode]] = {}

        for network_idx, nodes in composition.items():
            node_set = set(nodes)
            required = set()

            for node in nodes:
                for condition in conditions[node]:
                    if condition not in node_set:
                        required.add(condition)

            # remove duplicates
            networks[network_idx] = sort_nodes_topologically(self.simulation_graph, list(required))

        return networks

    def network_composition(self) -> dict[int, list[SimulationNode]]:
        """
        Returns a dictionary where the keys are integer network indices and the values
        are lists of nodes that are estimated by that inference network.
        """
        conditions = self.conditions_by_node()

        processed_nodes = set(k for k, v in conditions.items() if v == [])
        conditions = {k: v for k, v in conditions.items() if k not in processed_nodes}

        networks: dict[int, list[SimulationNode]] = {}
        network_idx = 0

        # Build inference layers iteratively: start with all nodes that require no conditions,
        # then repeatedly form the next layer by selecting nodes whose dependencies are entirely
        # covered by previous inference networks
        while conditions:
            networks[network_idx] = []
            next_nodeset = {k for k, v in conditions.items() if set(v).issubset(processed_nodes | set([k]))}

            if next_nodeset:
                processed_nodes.update(next_nodeset)

                for node in next_nodeset:
                    _ = conditions.pop(node)
                    networks[network_idx].extend([node])

            network_idx += 1

        for k, v in networks.items():
            networks[k] = sort_nodes_topologically(self.simulation_graph, list(set(v)))

        return networks

    def permutated_data_shape_order(self) -> list[SimulationNode]:
        """
        Return a permutation of `data_shape_order` suitable for summary network input.

        The returned list is reordered such that amortizable nodes appear first, followed
        by non-amortizable nodes, while preserving their relative order within each group.
        """
        shape_order = self.data_shape_order()
        amortizable = [n for n in shape_order if self.allows_amortization(n)]
        non_amortizable = [n for n in shape_order if not self.allows_amortization(n)]

        # put non amortizable nodes at the end
        return amortizable + non_amortizable

    def data_shape_order(self) -> list[SimulationNode]:
        """
        Determines the ordering of the data shape defined by the user-defined simulation graph.

        Returns a list of node names corresponding to the simulation dimensions of the data.
        Specifically, if the data shape is:

            (B, N_node_a, N_node_b, N_node_c, D)

        where `N_node_x` is the number of repetitions of that node during simulation, and
        `B` and `D` are the batch and data dimensions respectively, then this method returns:

            ['node_a', 'node_b', 'node_c']
        """

        merged = merge_root_nodes(self.simulation_graph)
        ordered = list(nx.lexicographical_topological_sort(merged))
        reps = self.simulation_graph.nodes[self.simulation_graph.data_node()]["reps"]

        if reps == 1:
            return ordered[1:-1]
        else:
            return ordered[1:]

    def amortizable_nodes(self) -> list[SimulationNode]:
        amortizable_nodes = []
        data_nodes = self.simulation_graph.data_node()

        for node in self.simulation_graph.nodes:
            if node not in data_nodes and self.allows_amortization(node):
                amortizable_nodes.append(node)

        return amortizable_nodes

    def allows_amortization(self, node: Node) -> bool:
        """
        Checks if a node in the simulation graph is amortizable,
        i.e., allows independent estimation of each group.
        """
        if node not in self.simulation_graph.nodes:
            raise ValueError(f"Node {node} not found.")

        conditions = self.detailed_conditions_by_node()
        node_names = self.original_node_names()
        data_node = self.simulation_graph.data_node()

        if node == data_node:
            return False

        for k, v in conditions.items():
            if node_names[k] == node:
                condition_names = [node_names[x] for x in v]
                if node in condition_names:
                    return False

        return True

    def original_node_names(self) -> dict[ExpandedNode, SimulationNode]:
        """
        Maps node names of the inverted graph to node names in the corresponding
        SimulationGraph.
        """
        mapping = {}

        for node in self.nodes:
            expanded_node = self.expanded_graph.nodes[node]

            if expanded_node["merged_from"] != []:
                merged_from = expanded_node["merged_from"]
                if len(merged_from) == 1:
                    mapping[node] = merged_from[0]
                else:
                    mapping[node] = merged_from
            elif expanded_node["previous_names"] == []:
                mapping[node] = node
            else:
                mapping[node] = expanded_node["previous_names"][0]

        return mapping

    def conditions_by_node(self) -> dict[SimulationNode, list[SimulationNode]]:
        """
        Same output as `detailed_conditions_by_node`, but uses original node
        names instead of names altered by graph expansion.
        """

        detailed_conditions = self.detailed_conditions_by_node()
        original_node_names = self.original_node_names()

        def names(node):
            if self.expanded_graph.nodes[node]["merged_from"]:
                return self.expanded_graph.nodes[node]["merged_from"]
            else:
                return [original_node_names[node]]

        result = {}

        for node, conditions in detailed_conditions.items():
            keys = names(node)

            values = set()

            for condition in conditions:
                node_names = names(condition)
                for name in node_names:
                    values.add(name)

            for k in keys:
                result[k] = sort_nodes_topologically(self.simulation_graph, list(values))

        return result

    def detailed_conditions_by_node(self) -> dict[ExpandedNode, list[ExpandedNode]]:
        """
        Returns a dictionary with nodes as keys and a list of nodes that directly precede
        it as values.
        """
        conditions = {node: [] for node in self.nodes}

        for node in nx.lexicographical_topological_sort(self):
            conditions[node] = list(self.predecessors(node))

        return conditions

    def get_config(self):
        graph_data = json_graph.node_link_data(self)

        config = {
            "graph_data": graph_data,
            "expanded_graph": self.expanded_graph,
        }

        return serialize(config)

    @classmethod
    def from_config(cls, config):
        graph_data = config["graph_data"]
        expanded_graph = ExpandedGraph.from_config(config["expanded_graph"]["config"])

        return cls(expanded_graph=expanded_graph, graph_data=graph_data)
