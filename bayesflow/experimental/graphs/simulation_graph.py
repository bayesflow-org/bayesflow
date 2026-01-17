import inspect
from copy import deepcopy
from typing import Any, Callable, TypeAlias

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .utils import (
    merge_root_nodes,
    sort_nodes_topologically,
    split_node,
    _retrieve_function_reference,
    _import_function_from_reference,
)

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


@serializable("bayesflow.experimental")
class SimulationGraph(nx.DiGraph):
    """
    Directed acyclic graph defining a simulation composed of sampling nodes.
    Created and mutated internally when building a `GraphicalSimulator`.

    A `SimulationGraph` is used to infer a factorization of the joint posterior
    in two stages: First, the graph is expanded into an `ExpandedGraph`.
    This is necessary to determine if variables from a node can be estimated
    group-wise (enabling amortization over groups).

    Second, an inversion algorithm is applied to the `ExpandedGraph` to produce
    an `InvertedGraph`, which served as an input to the `GraphicalApproximator`.

    Parameters
    ----------
    meta_fn : Callable[[], dict[str, Any]] | None
        Function returning a dict of meta data.
        This meta data can be used to dynamically vary the number of sampling repetitions (`reps`)
        for nodes added via `add_node`.
    """

    def __init__(self, meta_fn: Callable | None = None, graph_data=None, **kwargs):
        super().__init__(**kwargs)
        self.meta_fn = meta_fn

        if graph_data is not None:
            g = json_graph.node_link_graph(graph_data, directed=True, multigraph=False)

            self.add_nodes_from(g.nodes(data=True))
            self.add_edges_from(g.edges(data=True))

    def expand(self, merge_roots: bool = True):
        """
        Expands the graph by splitting interior nodes into explicit subgraphs.

        Returns
        -------
        ExpandedGraph
            Expanded representation of the simulation graph.
        """
        from .expanded_graph import ExpandedGraph

        graph = deepcopy(self)
        if merge_roots:
            graph = merge_root_nodes(graph)

        for node in nx.lexicographical_topological_sort(graph):
            interior_node = graph.in_degree(node) != 0 and graph.out_degree(node) != 0

            if interior_node and node in graph.nodes:
                graph = split_node(graph, node)

        for node in nx.lexicographical_topological_sort(graph):
            for key in ["split_by", "previous_names", "merged_from"]:
                if key not in graph.nodes[node]:
                    graph.nodes[node][key] = []

        return ExpandedGraph(simulation_graph=self, graph_data=json_graph.node_link_data(graph))

    def invert(self, merge_roots: bool = True):
        """
        Inverts the expanded simulation graph.

        Parameters
        ----------
        merge_roots : bool, optional
            Whether to merge root nodes in the inverted graph.

        Returns
        -------
        InvertedGraph
            Inverted representation of the expanded graph.
        """
        expanded_graph = self.expand(merge_roots=merge_roots)
        inverted_graph = expanded_graph.invert(merge_roots=merge_roots)

        return inverted_graph

    def variable_names(self) -> dict[SimulationNode, list[str]]:
        """
        Returns a mapping from each node to the list of variable names it produces.

        The graph is evaluated once in topological order to collect sample outputs.
        This may be expensive; results are cached in `GraphicalApproximator`.
        """

        def _call_sample_fn(sample_fn: Callable[[], dict[str, Any]], args) -> dict[str, Any]:
            signature = inspect.signature(sample_fn)
            fn_args = signature.parameters
            accepted_args = {k: v for k, v in args.items() if k in fn_args}

            return sample_fn(**accepted_args)

        simulation_graph = deepcopy(self)
        meta_dict = simulation_graph.meta_fn() if simulation_graph.meta_fn else {}
        samples_by_node = {}

        for node in nx.lexicographical_topological_sort(simulation_graph):
            simulation_graph.nodes[node]["reps"] = 1
            parent_nodes = list(simulation_graph.predecessors(node))
            sample_fn = simulation_graph.nodes[node]["sample_fn"]

            if not parent_nodes:
                samples_by_node[node] = _call_sample_fn(sample_fn, {})
            else:
                parent_samples = [samples_by_node[p] for p in parent_nodes]
                merged_dict = {k: v for d in parent_samples for k, v in d.items()}

                sample_fn_input = merged_dict | meta_dict
                samples_by_node[node] = _call_sample_fn(sample_fn, sample_fn_input)

        return {k: list(v.keys()) for k, v in samples_by_node.items()}

    def output_shapes(self, meta_dict: dict | None = None):
        """
        Returns the output shape of each simulated variable in the simulation graph.

        The graph is evaluated once in topological order to collect sample outputs.
        This may be expensive; results are cached in `GraphicalApproximator`.
        _______
        """
        simulation_graph = deepcopy(self)
        variable_names = self.variable_names()
        output_dimensions = self.output_dimensions(meta_dict=meta_dict)

        output_shapes = {}

        for node in nx.lexicographical_topological_sort(simulation_graph):
            ancestors = nx.ancestors(self, node)
            sorted_ancestors = sort_nodes_topologically(self, [n for n in ancestors])

            reps = [self.nodes[n]["reps"] for n in sorted_ancestors]
            reps.append(self.nodes[node]["reps"])

            for variable_name in variable_names[node]:
                output_shapes[variable_name] = ["B"]
                for rep in reps:
                    if rep != 1:
                        output_shapes[variable_name].append(rep)

                output_shapes[variable_name].extend(output_dimensions[variable_name])

        for k, v in output_shapes.items():
            if meta_dict:
                v = [meta_dict.get(x, x) for x in v]

            output_shapes[k] = tuple(v)

        return output_shapes

    def output_dimensions(self, meta_dict: dict | None = None):
        """
        Returns the output dimension of each simulated variable in the simulation graph.

        The graph is evaluated once in topological order to collect sample outputs.
        This may be expensive; results are cached in `GraphicalApproximator`.
        _______
        """

        def _call_sample_fn(sample_fn: Callable[[], dict[str, Any]], args) -> dict[str, Any]:
            signature = inspect.signature(sample_fn)
            fn_args = signature.parameters
            accepted_args = {k: v for k, v in args.items() if k in fn_args}

            return sample_fn(**accepted_args)

        simulation_graph = deepcopy(self)
        if not meta_dict:
            meta_dict = simulation_graph.meta_fn() if simulation_graph.meta_fn else {}

        samples_by_node = {}
        output_dimensions = {}

        for node in nx.lexicographical_topological_sort(simulation_graph):
            simulation_graph.nodes[node]["reps"] = 1  # settings reps to 1 for computational efficiency
            parent_nodes = list(simulation_graph.predecessors(node))
            sample_fn = simulation_graph.nodes[node]["sample_fn"]

            if not parent_nodes:
                samples_by_node[node] = _call_sample_fn(sample_fn, {})
            else:
                parent_samples = [samples_by_node[p] for p in parent_nodes]
                merged_dict = {k: v for d in parent_samples for k, v in d.items()}

                sample_fn_input = merged_dict | meta_dict
                samples_by_node[node] = _call_sample_fn(sample_fn, sample_fn_input)

            for variable_name, value in samples_by_node[node].items():
                output_dimensions[variable_name] = np.shape(np.atleast_1d(value))

        return output_dimensions

    def data_node(self) -> SimulationNode:
        """
        Returns the terminal (leaf) node of the simulation graph.

        Returns
        -------
        SimulationNode
            Node with no outgoing edges.
        """
        leaf_nodes = [n for n, d in self.out_degree() if d == 0]

        return leaf_nodes[0]

    def get_config(self):
        graph_data = json_graph.node_link_data(self)

        for node in graph_data["nodes"]:
            if "sample_fn" in node and node["sample_fn"] is not None:
                node["sample_fn_ref"] = _retrieve_function_reference(node["sample_fn"])
                del node["sample_fn"]

        if self.meta_fn is not None:
            meta_fn_ref = _retrieve_function_reference(self.meta_fn)
        else:
            meta_fn_ref = None

        config = {"graph_data": graph_data, "meta_fn_ref": meta_fn_ref}
        return serialize(config)

    @classmethod
    def from_config(cls, config):
        graph_data = config["graph_data"]

        for node in graph_data["nodes"]:
            if "sample_fn_ref" in node and node["sample_fn_ref"] is not None:
                node["sample_fn"] = _import_function_from_reference(node["sample_fn_ref"])
                del node["sample_fn_ref"]

        if config["meta_fn_ref"] is not None:
            meta_fn = _import_function_from_reference(config["meta_fn_ref"])
        else:
            meta_fn = None

        return cls(meta_fn=meta_fn, graph_data=graph_data)
