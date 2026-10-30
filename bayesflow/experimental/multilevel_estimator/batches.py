import itertools
import math
from typing import Callable

import keras
import networkx as nx
import numpy as np

from bayesflow.adapters import Adapter

from ..multilevel_simulator import MultilevelSimulator
from ..multilevel_simulator.indexing_helpers import concatenated_ranges, group_ids
from ..multilevel_simulator.samples import GraphSamples, num_datasets, select
from ..graphs import InvertedGraph, SimulationGraph
from ..graphs.simulation_graph import is_exchangeable, is_interior
from .encoder import gather, segment_mean


class SimulationDataset(keras.utils.PyDataset):
    def __init__(
        self,
        source: MultilevelSimulator | GraphSamples,
        batch_size: int,
        to_batch: Callable[[GraphSamples], dict],
        num_batches: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.source = source
        self.batch_size = batch_size
        self.to_batch = to_batch

        if isinstance(source, MultilevelSimulator):
            self.length = num_batches
        else:
            datasets = num_datasets(source)
            self.length = math.ceil(datasets / batch_size)
            self.order = np.random.permutation(datasets)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> dict:
        if isinstance(self.source, MultilevelSimulator):
            simulation = self.source.sample(self.batch_size)
            return self.to_batch(simulation, padded=True)

        datasets = self.order[i * self.batch_size : (i + 1) * self.batch_size]
        simulation = select(self.source, datasets)

        return self.to_batch(simulation, padded=True)

    def on_epoch_end(self) -> None:
        if not isinstance(self.source, MultilevelSimulator):
            self.order = np.random.permutation(num_datasets(self.source))


def to_batch(samples: GraphSamples, graph: InvertedGraph, adapter: Adapter, padded: bool = False) -> dict:
    # Adds groups and steps to the samples and runs the adapter.
    # This is the input that inference networks use
    simulation_graph = graph.expanded_graph.simulation_graph

    batch = {}
    for node, node_samples in samples.items():
        batch[node] = {"indices": node_samples.indices, "values": node_samples.values}

        if simulation_graph.out_degree(node) == 0:
            batch[node]["groups"] = groups(simulation_graph, node, node_samples.indices)
        elif not graph.is_amortizable(node):
            batch[node]["steps"] = steps(node, node_samples.indices)

    if padded:
        batch = pad(batch)

    for entries in batch.values():
        entries["values"] = adapter(entries["values"], strict=False)

    return batch


def groups(graph: SimulationGraph, data_node: str, indices: dict[str, np.ndarray]) -> dict[str, dict]:
    # Helper function for to_batch.
    # For each grouping of exchangeable factors, compute to which group the row belongs to.
    result = {}

    for factors in groupings(graph, data_node):
        columns = []
        for factor in ["dataset", *factors]:
            columns.append(indices[factor])

        ids = group_ids(columns)
        name = "+".join(factors) or "dataset"
        result[name] = {"ids": ids, "counts": np.bincount(ids)}

    return result


def groupings(graph: SimulationGraph, data_node: str) -> list[tuple[str, ...]]:
    # Lists the different ways that rows of the data node can be grouped by.
    # This is necessary for summarizing the data.
    ancestors = nx.ancestors(graph, data_node)

    grouping_factors = []
    for node in nx.lexicographical_topological_sort(graph):
        if node in ancestors and is_interior(graph, node) and is_exchangeable(graph, node):
            grouping_factors.append(node)

    valid = []
    for size in range(len(grouping_factors) + 1):
        for factors in itertools.combinations(grouping_factors, size):
            outer_factors = set()
            for factor in factors:
                outer_factors |= nx.ancestors(graph, factor) & set(grouping_factors)

            if outer_factors <= set(factors):
                valid.append(factors)

    if not is_exchangeable(graph, data_node):
        valid.remove(tuple(grouping_factors))

    return valid


def steps(node: str, indices: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Helper function for to_batch.
    # Computes for each row its sequence and position, as well as all (earlier, later)
    # row pairs within a sequence. Used for sequentially inferred nodes.
    columns = []
    for column, ids in indices.items():
        if column != node:
            columns.append(ids)

    ids = group_ids(columns)
    counts = np.bincount(ids)
    starts = np.cumsum(counts) - counts
    order = np.argsort(ids, kind="stable")

    positions = np.empty(len(ids), dtype=np.int64)
    positions[order] = np.arange(len(ids)) - np.repeat(starts, counts)

    later = np.repeat(np.arange(len(ids)), positions)
    earlier = order[concatenated_ranges(starts[ids], positions)]

    return {"ids": ids, "counts": counts, "positions": positions, "earlier": earlier, "later": later}


def pad(batch: dict) -> dict:
    # Pad output of to_batch, so that inference networks do not see so many
    # dataset sizes, which would trigger a long compilation run
    sizes = {}
    datasets = 0

    for node, entries in batch.items():
        sizes[node] = bucket(len(entries["indices"][node]))
        datasets = max(datasets, int(np.max(entries["indices"]["dataset"])) + 1)

    sizes["dataset"] = bucket(datasets)

    for node, entries in batch.items():
        rows = len(entries["indices"][node])
        size = sizes[node]

        indices = {}
        for column, ids in entries["indices"].items():
            indices[column] = pad_with(ids, size, sizes[column] - 1)

        values = {}
        for variable, value in entries["values"].items():
            values[variable] = repeat_rows(value, size)

        weights = np.full(rows, size / rows, dtype=keras.config.floatx())

        entries["indices"] = indices
        entries["values"] = values
        entries["weights"] = pad_with(weights, size, 0.0)

        numberings = list(entries.get("groups", {}).values())
        if "steps" in entries:
            numberings.append(entries["steps"])

        # ids and counts of groups and sequences are padded alike
        for group in numberings:
            num_groups = bucket(len(group["counts"]))
            group["ids"] = pad_with(group["ids"], size, num_groups - 1)
            group["counts"] = pad_with(group["counts"], num_groups, 0)

        if "steps" in entries:
            steps = entries["steps"]
            num_pairs = bucket(len(steps["earlier"]))
            steps["positions"] = pad_with(steps["positions"], size, 0)
            steps["earlier"] = pad_with(steps["earlier"], num_pairs, size - 1)
            steps["later"] = pad_with(steps["later"], num_pairs, size - 1)

    return batch


def bucket(n: int) -> int:
    # the next of four sizes per doubling that is larger than n
    return int(np.ceil(2 ** (np.ceil(4 * np.log2(n + 1)) / 4)))


def pad_with(array, size: int, value):
    array = np.asarray(array)
    shape = (size - len(array), *array.shape[1:])
    padding = np.full(shape, value, dtype=array.dtype)

    return np.concatenate([array, padding])


def repeat_rows(values, size: int):
    return np.asarray(values)[np.arange(size) % len(values)]


def add_sequence_summaries(batch: dict, node: str, conditions):
    # Used for constructing summaries for sequentially inferred nodes.
    # Adds the mean conditions of the sequence, and the mean and number of the earlier values
    steps = batch[node]["steps"]
    values = node_values(batch, node)

    num_sequences = keras.ops.shape(steps["counts"])[0]
    sequence_mean = segment_mean(conditions, steps["ids"], num_sequences)
    sequence_conditions = gather(sequence_mean, steps["ids"])

    num_rows = keras.ops.shape(values)[0]
    earlier_values = gather(values, steps["earlier"])
    earlier_sum = keras.ops.segment_sum(earlier_values, steps["later"], num_segments=num_rows)

    num_earlier = keras.ops.cast(steps["positions"], values.dtype)[:, None]
    earlier_mean = earlier_sum / keras.ops.maximum(num_earlier, 1.0)
    log_num_earlier = keras.ops.log1p(num_earlier)

    return keras.ops.concatenate([conditions, sequence_conditions, earlier_mean, log_num_earlier], axis=-1)


def network_variables(batch: dict, target_nodes: list[str]):
    values = []

    for node in target_nodes:
        values.append(node_values(batch, node))

    return keras.ops.concatenate(values, axis=-1)


def node_values(batch: dict, node: str):
    values = []

    for variable in sorted(batch[node]["values"]):
        values.append(batch[node]["values"][variable])

    concatenated = keras.ops.concatenate(values, axis=-1)

    return keras.ops.cast(concatenated, keras.config.floatx())
