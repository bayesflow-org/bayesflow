import inspect
import itertools
from collections.abc import Callable
from typing import Any, Optional

import networkx as nx
import numpy as np

from bayesflow.simulators import Simulator
from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size


class GraphicalSimulator(Simulator):
    """
    A graph-based simulator that generates samples by traversing a DAG
    and calling user-defined sample functions at each node.

    Parameters
    ----------
    meta_fn : Optional[Callable[[], dict[str, Any]]]
        A callable that returns a dictionary of meta data.
        This meta data can be used to dynamically vary the number of sample repetitions (`reps`)
        for nodes added via `add_node`.
    """

    def __init__(self, meta_fn: Optional[Callable[[], dict[str, Any]]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = nx.DiGraph()
        self.meta_fn = meta_fn

    def add_node(self, node: str, sample_fn: Callable[..., dict[str, Any]], reps: int | str = 1):
        self.graph.add_node(node, sample_fn=sample_fn, reps=reps)

    def add_edge(self, from_node: str, to_node: str):
        self.graph.add_edge(from_node, to_node)

    @allow_batch_size
    def sample(self, batch_shape: Shape | int, **kwargs) -> dict[str, np.ndarray]:
        """
        Generates samples by topologically traversing the DAG.
        For each node, the sample function is called based on parent values.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number of samples,
            but an int can also be passed.
        **kwargs
            Currently unused
        """
        _ = kwargs  # Simulator class requires **kwargs, which are unused here
        meta_dict = self.meta_fn() if self.meta_fn else {}
        samples_by_node = {}

        # Initialize samples container for each node
        for node in self.graph.nodes:
            samples_by_node[node] = np.empty(batch_shape, dtype="object")

        ordered_nodes = list(nx.topological_sort(self.graph))

        for batch_idx in np.ndindex(batch_shape):
            for node in ordered_nodes:
                node_samples = []

                parent_nodes = list(self.graph.predecessors(node))
                sample_fn = self.graph.nodes[node]["sample_fn"]
                reps_field = self.graph.nodes[node]["reps"]
                reps = reps_field if isinstance(reps_field, int) else meta_dict[reps_field]

                if not parent_nodes:
                    # root node: generate independent samples
                    node_samples = [
                        {"__batch_idx": batch_idx, f"__{node}_idx": i} | self._call_sample_fn(sample_fn, {})
                        for i in range(reps)
                    ]
                else:
                    # non-root node: depends on parent samples
                    parent_samples = [samples_by_node[p][batch_idx] for p in parent_nodes]
                    merged_dicts = merge_lists_of_dicts(parent_samples)

                    for merged in merged_dicts:
                        index_entries = {k: v for k, v in merged.items() if k.startswith("__")}
                        variable_entries = {k: v for k, v in merged.items() if not k.startswith("__")}

                        sample_fn_input = variable_entries | meta_dict
                        samples = [
                            index_entries | {f"__{node}_idx": i} | self._call_sample_fn(sample_fn, sample_fn_input)
                            for i in range(reps)
                        ]
                        node_samples.extend(samples)

                samples_by_node[node][batch_idx] = node_samples

        # collect outputs
        output_dict = {}
        for node in nx.topological_sort(self.graph):
            output_dict.update(self._collect_output(samples_by_node[node]))

        output_dict.update(meta_dict)

        return output_dict

    def _collect_output(self, samples):
        output_dict = {}

        # retrieve node and ancestors from internal sample representation
        index_entries = [k for k in samples.flat[0][0].keys() if k.startswith("__")]
        node = index_entries[-1].removeprefix("__").removesuffix("_idx")
        ancestors = sorted_ancestors(self.graph, node)

        # build dict of node repetitions
        reps = {}
        for ancestor in ancestors:
            reps[ancestor] = max(s[f"__{ancestor}_idx"] for s in samples.flat[0]) + 1
        reps[node] = max(s[f"__{node}_idx"] for s in samples.flat[0]) + 1

        variable_names = self._variable_names(samples)

        # collect output for each variable
        for variable in variable_names:
            output_shape = self._output_shape(samples, variable)
            output_dict[variable] = np.empty(output_shape)

            for batch_idx in np.ndindex(samples.shape):
                for sample in samples[batch_idx]:
                    idx = [*batch_idx]

                    # add index elements for ancestors
                    for ancestor in ancestors:
                        if reps[ancestor] != 1:
                            idx.append(sample[f"__{ancestor}_idx"])

                    # add index elements for node
                    if reps[node] != 1:
                        idx.append(sample[f"__{node}_idx"])

                    output_dict[variable][tuple(idx)] = sample[variable]

        return output_dict

    def _variable_names(self, samples):
        return [k for k in samples.flat[0][0].keys() if not k.startswith("__")]

    def _output_shape(self, samples, variable):
        index_entries = [k for k in samples.flat[0][0].keys() if k.startswith("__")]
        node = index_entries[-1].removeprefix("__").removesuffix("_idx")

        # start with batch shape
        batch_shape = samples.shape
        output_shape = list(batch_shape)
        ancestors = sorted_ancestors(self.graph, node)

        # add ancestor reps
        for ancestor in ancestors:
            node_reps = max(s[f"__{ancestor}_idx"] for s in samples.flat[0]) + 1
            if node_reps != 1:
                output_shape.append(node_reps)

        # add node reps
        node_reps = max(s[f"__{node}_idx"] for s in samples.flat[0]) + 1
        if node_reps != 1:
            output_shape.append(node_reps)

        # add variable shape
        variable_shape = np.atleast_1d(samples.flat[0][0][variable]).shape
        output_shape.extend(variable_shape)

        return tuple(output_shape)

    def _call_sample_fn(self, sample_fn, args):
        signature = inspect.signature(sample_fn)
        fn_args = signature.parameters
        accepted_args = {k: v for k, v in args.items() if k in fn_args}

        return sample_fn(**accepted_args)


def sorted_ancestors(graph, node):
    return [n for n in nx.topological_sort(graph) if n in nx.ancestors(graph, node)]


def merge_lists_of_dicts(nested_list: list[list[dict]]) -> list[dict]:
    """
    Merges all combinations of dictionaries from a list of lists.
    Equivalent to a Cartesian product of dicts, then flattening.

    Examples:
        >>> merge_lists_of_dicts([[{"a": 1, "b": 2}], [{"c": 3}, {"d": 4}]])
        [{'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2, 'd': 4}]
    """

    all_combinations = itertools.product(*nested_list)
    return [{k: v for d in combo for k, v in d.items()} for combo in all_combinations]
