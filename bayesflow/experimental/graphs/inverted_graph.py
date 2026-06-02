from copy import deepcopy
from typing import Literal, NamedTuple, TypeAlias

import networkx as nx
import sympy as sp
from networkx.readwrite import json_graph

from bayesflow.utils.serialization import serializable, serialize

from .expanded_graph import ExpandedGraph
from .utils import merge_root_nodes, sort_nodes_topologically

Node: TypeAlias = str
SimulationNode: TypeAlias = str
ExpandedNode: TypeAlias = str


class SummaryKey(NamedTuple):
    """
    Used internally to identify a summary network with a combination of mode, inferred_nodes and chain_step.

    For ``"global"`` mode, ``inferred_nodes`` is a tuple of all inferred nodes that share
    this summary.
    For ``"per_level"`` mode, ``inferred_nodes`` is a 1-tuple identifying which level is kept
    non-flattened, so different inferred nodes at different levels get
    different summary networks.
    For ``"non_exchangeable"`` mode, the network summarizes previously sampled values
    of ``conditioned_node`` to condition the next sample in the NonExchangeableWrapper.
    ``chain_step`` identifies the position in the sequential summary chain.
    """

    conditioned_node: SimulationNode
    mode: Literal["global", "per_level", "non_exchangeable"]
    inferred_nodes: tuple[SimulationNode, ...]
    chain_step: int = 0


@serializable("bayesflow.experimental")  # type: ignore[missing-argument]
class InvertedGraph(nx.DiGraph):
    """
    Directed graph representing the factorization of the joint posterior of a
    forward model defined by `SimulationGraph`.

    An `InvertedGraph` is derived from an `ExpandedGraph` retrieved by calling the `expand()`
    method on a `SimulationGraph`.
    """

    def __init__(self, *, expanded_graph: ExpandedGraph, graph_data=None, **kwargs):
        super().__init__(**kwargs)  # optionally initializing with existing data

        self.simulation_graph = deepcopy(expanded_graph.simulation_graph)
        self.expanded_graph = deepcopy(expanded_graph)

        if graph_data is not None:
            g = json_graph.node_link_graph(graph_data, directed=True, multigraph=False)

            self.add_nodes_from(g.nodes(data=True))
            self.add_edges_from(g.edges(data=True))

    def required_summary_networks(self) -> None:
        """
        Returns the required summary networks for the `GraphicalApproximator`.
        """
        shapes = self.summary_network_input_shapes()
        mode_labels = {
            "global": "(global summary)",
            "per_level": "(per-level summary)",
            "non_exchangeable": "(for non-exchangeable inferred node)",
        }

        print(f"Summary Networks ({len(shapes)} total)")
        print("_" * len(f"Summary Networks ({len(shapes)} total)"))
        print("")

        for i, (key, input_shape) in enumerate(shapes.items()):
            output_shape = input_shape[:-2] + (sp.Symbol(f"summary_dim_{i}"),)
            inferred = ", ".join(key.inferred_nodes)
            label = mode_labels[key.mode]

            if key.mode != "non_exchangeable":
                print(f"{i}: Summary of {key.conditioned_node} {label} for {inferred}")
            else:
                print(f"{i}: Summary of {key.conditioned_node} {label}")

            print(f"{' ' * (len(str(i)) + 3)} Input shape:  {input_shape}")
            print(f"{' ' * (len(str(i)) + 3)} Output shape: {output_shape}")
            print("")

    def required_inference_networks(self) -> None:
        """
        Returns the required inference networks for the `GraphicalApproximator`.
        """
        composition = self.network_composition()
        conditions = self.network_conditions()
        variable_shapes = self.inference_variable_shapes()
        condition_shapes = self.inference_condition_shapes()

        print(f"Inference Networks ({len(composition)} total)")
        print("_" * len(f"Inference Networks ({len(composition)} total)"))
        print("")

        for i in composition.keys():
            inferred = ", ".join(composition[i])
            conditioned = ", ".join(conditions[i])

            print(f"{i}: infers {inferred} from {conditioned}")

            print(f"{' ' * (len(str(i)) + 3)} Output shape:     {variable_shapes[i]}")
            print(f"{' ' * (len(str(i)) + 3)} Conditions shape: {condition_shapes[i]}")
            print("")

    def num_summary_networks(self) -> int:
        """
        Returns the number of required summary networks for the `GraphicalApproximator`.
        """
        return max(1, len(self.data_shape_order()))

    def inference_variable_shapes(self) -> dict[int, tuple]:
        """
        Returns the symbolic shape of the concatenated inference variable tensor
        for each inference network.

        Returns
        -------
        dict[int, tuple]
            Mapping from network index to symbolic shape tuple.
        """
        output_shapes = self.simulation_graph.output_shapes()
        variable_names = self.simulation_graph.variable_names()

        result = {}
        for network_idx, nodes in self.network_composition().items():
            shapes = []
            for node in nodes:
                for var in variable_names[node]:
                    if var in output_shapes:
                        shapes.append(output_shapes[var])

            if shapes:
                prefix = shapes[0][:-1]
                total_D = sum(s[-1] for s in shapes)
                result[network_idx] = prefix + (total_D,)

        return result

    def inference_condition_shapes(self) -> dict[int, tuple]:
        """
        Returns the symbolic shape of the inference condition tensor for each inference network.

        Returns
        -------
        dict[int, tuple]
            Mapping from network index to symbolic shape tuple.
        """
        output_shapes = self.simulation_graph.output_shapes()
        variable_names = self.simulation_graph.variable_names()
        network_composition = self.network_composition()
        network_conditions = self.network_conditions()

        global_inferred = self._global_summary_inferred_nodes()
        summary_dims = {key: sp.Symbol(f"summary_dim_{i}") for i, key in enumerate(self.summary_network_input_shapes())}

        data_node = self.simulation_graph.data_node()
        n_node_reps = len(output_shapes[variable_names[data_node][0]]) - 2

        result = {}

        for network_idx, inferred_nodes in network_composition.items():
            inferred_prefix = output_shapes[variable_names[inferred_nodes[0]][0]][:-1]
            total_feature_dim = 0

            for conditioned_node in network_conditions[network_idx]:
                conditioned_vars = variable_names[conditioned_node]
                total_D = sum(output_shapes[v][-1] for v in conditioned_vars)
                source_shape = output_shapes[conditioned_vars[0]]

                shared_prefix = [dim for dim in inferred_prefix if dim in source_shape]
                k = len(source_shape) - 1 - len(shared_prefix)

                if k <= 0:
                    total_feature_dim += total_D
                else:
                    mode = "per_level" if self.is_per_level_summary(inferred_nodes[0], conditioned_node) else "global"
                    key = SummaryKey(
                        conditioned_node=conditioned_node,
                        mode=mode,
                        inferred_nodes=(inferred_nodes[0],)
                        if mode == "per_level"
                        else global_inferred[conditioned_node],
                        chain_step=k - 1,
                    )
                    total_feature_dim += summary_dims[key]

            result[network_idx] = inferred_prefix + (total_feature_dim + n_node_reps,)

        return result

    def non_amortizable_summary_input_shapes(self) -> dict[int, tuple]:
        """
        Returns the symbolic input shape for the sequential summary network
        required by each non-amortizable inference network.

        A non-amortizable network needs an auto-regressive summary network
        that is fed the previously sampled values of the same node. Its input
        shape therefore matches the inference variable shape of that network.

        Returns
        -------
        dict[int, tuple]
            Mapping from network index to symbolic input shape, ordered by
            network index. Empty if all nodes are amortizable.
        """
        var_shapes = self.inference_variable_shapes()
        result = {}
        for network_idx, nodes in self.network_composition().items():
            if any(not self.allows_amortization(node) for node in nodes):
                result[network_idx] = var_shapes[network_idx]

        return result

    def _global_summary_inferred_nodes(self) -> dict[str, tuple[str, ...]]:
        """
        Returns a joint inferred_nodes tuple for each conditioned node that requires a global summary.
        """
        network_composition = self.network_composition()
        network_conditions = self.network_conditions()
        result = {}
        for network_idx, inferred_nodes in network_composition.items():
            for conditioned_node in network_conditions[network_idx]:
                if not self.is_per_level_summary(inferred_nodes[0], conditioned_node):
                    existing = set(result.get(conditioned_node, ()))
                    existing.update(inferred_nodes)
                    result[conditioned_node] = tuple(sorted(existing))

        return result

    def summary_network_input_shapes(self) -> dict["SummaryKey", tuple]:
        """
        Returns an ordered mapping from ``SummaryKey`` to symbolic input shape for
        each summary network required by this graph.
        """
        output_shapes = self.simulation_graph.output_shapes()
        variable_names = self.simulation_graph.variable_names()
        network_composition = self.network_composition()
        network_conditions = self.network_conditions()

        global_inferred = self._global_summary_inferred_nodes()
        result = {}

        for network_idx, inferred_nodes in network_composition.items():
            first_vars = variable_names[inferred_nodes[0]]
            inferred_prefix = output_shapes[first_vars[0]][:-1]

            for conditioned_node in network_conditions[network_idx]:
                conditioned_vars = variable_names[conditioned_node]
                conditioned_prefix = output_shapes[conditioned_vars[0]][:-1]
                total_D = sum(output_shapes[v][-1] for v in conditioned_vars)
                conditioned_shape = conditioned_prefix + (total_D,)

                # permute to align with inferred_prefix, skipping dims not present
                source = list(conditioned_shape)
                shared_prefix = [dim for dim in inferred_prefix if dim in source]
                prefix_indices = [source.index(dim) for dim in shared_prefix]
                remaining_indices = [i for i in range(len(source)) if i not in prefix_indices]
                perm = prefix_indices + remaining_indices
                conditioned_shape = tuple(conditioned_shape[i] for i in perm)

                if len(conditioned_shape) <= len(shared_prefix) + 1:
                    continue

                if self.is_per_level_summary(inferred_nodes[0], conditioned_node):
                    mode = "per_level"
                else:
                    mode = "global"

                input_shape = conditioned_shape
                n_extra = len(conditioned_shape[1:-1]) - len(shared_prefix) + 1

                for step in range(n_extra):
                    key = SummaryKey(
                        conditioned_node=conditioned_node,
                        mode=mode,
                        inferred_nodes=(inferred_nodes[0],)
                        if mode == "per_level"
                        else global_inferred[conditioned_node],
                        chain_step=step,
                    )
                    idx = len(result)
                    if key not in result:
                        result[key] = input_shape

                    input_shape = input_shape[:-2] + (f"summary_dim_{idx}",)

        # append sequential summary networks for non-amortizable nodes
        for network_idx, nodes in network_composition.items():
            if any(not self.allows_amortization(node) for node in nodes):
                key = SummaryKey(conditioned_node=nodes[0], mode="non_exchangeable", inferred_nodes=tuple(nodes))
                if key not in result:
                    result[key] = self.inference_variable_shapes()[network_idx]

        return result

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

    def is_per_level_summary(self, inferred_node: SimulationNode, conditioned_node: SimulationNode) -> bool:
        """
        Returns True if ``conditioned_node`` requires a per-level summary when
        conditioning ``inferred_node``, False if a global summary is required.
        """
        detailed = self.detailed_conditions_by_node()
        original_names = self.original_node_names()

        per_copy_conditioned: dict = {}

        for expanded_inferred, conditions in detailed.items():
            if original_names[expanded_inferred] != inferred_node:
                continue

            expanded_conditioned = frozenset(c for c in conditions if original_names[c] == conditioned_node)
            if not expanded_conditioned:
                continue

            per_copy_conditioned[expanded_inferred] = expanded_conditioned

        if len(per_copy_conditioned) <= 1:
            return False

        all_sets = list(per_copy_conditioned.values())
        for i in range(len(all_sets)):
            for j in range(i + 1, len(all_sets)):
                if all_sets[i] & all_sets[j]:
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
