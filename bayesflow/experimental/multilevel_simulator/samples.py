from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from .indexing_helpers import concatenated_ranges


@dataclass
class NodeSamples:
    """
    The samples of one node in long format.

    Every row holds one sample of the node, and the rows are sorted by dataset.

    Attributes
    ----------
    indices : dict[str, np.ndarray]
        Integer index columns of shape `(num_rows,)`: ``"dataset"`` with the
        dataset of each row, one column per ancestor with the row of that
        ancestor the sample belongs to, and one column named after the node
        with the row's own position. Ids count rows of all datasets, not
        within each dataset.
    values : dict[str, np.ndarray]
        The samples of each variable of the node, with shape `(num_rows, dim)`.
    """

    indices: dict[str, np.ndarray]
    values: dict[str, np.ndarray]


#: The samples of every node of the graph, keyed by node name.
GraphSamples: TypeAlias = dict[str, NodeSamples]


def to_tensors(samples: GraphSamples, datasets=None, nodes: list[str] | None = None) -> dict[str, np.ndarray]:
    """
    Stacks the datasets of the samples into one array per variable.

    This is the format the diagnostics and most plotting code expect. It only
    works if all datasets have the same structure, so for models where the
    number of groups varies, convert one dataset at a time with `datasets`.

    Parameters
    ----------
    samples : GraphSamples
        Samples of every node, such as the output of the MultilevelSimulator or
        the posterior samples of the MultilevelEstimator.
    datasets : int or np.ndarray, optional
        Only convert these datasets (default is all datasets).
    nodes : list[str], optional
        Only convert the variables of these nodes (default is all nodes).

    Returns
    -------
    dict[str, np.ndarray]
        One array per variable. Variables of root nodes have shape
        `(num_datasets, dim)`, all others `(num_datasets, num_groups, dim)`.
        Posterior samples have an extra axis of size `num_samples` after
        the dataset axis.

    Raises
    ------
    ValueError
        If the datasets differ in the structure of a node.
    """

    if datasets is not None:
        samples = select(samples, np.atleast_1d(datasets))

    if nodes is None:
        nodes = list(samples)

    count = num_datasets(samples)
    tensors = {}

    for node in nodes:
        node_samples = samples[node]

        if not has_same_structure(node_samples.indices, count):
            raise ValueError(
                f"The datasets differ in the structure of node '{node}', so its samples cannot be stacked. "
                "Use the long format, or select datasets with the same structure, e.g. datasets=[0]."
            )

        is_root = len(node_samples.indices) == 2

        for variable, value in node_samples.values.items():
            value = value.reshape(count, -1, *value.shape[1:])

            if is_root:
                tensors[variable] = value[:, 0]
            else:
                tensors[variable] = np.moveaxis(value, 1, -2)

    return tensors


def has_same_structure(indices: dict[str, np.ndarray], datasets: int) -> bool:
    rows_per_dataset = np.bincount(indices["dataset"], minlength=datasets)

    if np.any(rows_per_dataset != rows_per_dataset[0]):
        return False

    for ids in indices.values():
        ids = ids.reshape(datasets, -1)
        relative_ids = ids - np.min(ids, axis=1, keepdims=True)

        if np.any(relative_ids != relative_ids[0]):
            return False

    return True


def num_rows(indices: dict[str, np.ndarray]) -> int:
    """
    Counts the rows of a node.

    Parameters
    ----------
    indices : dict[str, np.ndarray]
        The index columns of the node, as in :py:attr:`NodeSamples.indices`

    Returns
    -------
    int
        The number of rows.
    """
    return len(indices["dataset"])


def num_datasets(samples: GraphSamples) -> int:
    """
    Counts the datasets in the samples.

    Parameters
    ----------
    samples : GraphSamples
        The samples of every node.

    Returns
    -------
    int
        The number of datasets.
    """
    return 1 + max(int(np.max(node_samples.indices["dataset"])) for node_samples in samples.values())


def concatenate(all_samples: list[GraphSamples]) -> GraphSamples:
    """
    Combines several samples into one.

    The datasets of later samples are numbered after those of earlier ones.
    This also shifts the index columns, so that every index is consecutive.

    Parameters
    ----------
    all_samples : list[GraphSamples]
        Samples of the same graph, with the same nodes and variables.

    Returns
    -------
    GraphSamples
        Samples holding the datasets of all samples.
    """
    # Concatenate the samples by shifting all indices past those of the previous samples.

    concatenated = {}

    for node, first in all_samples[0].items():
        indices = {}

        for column in first.indices:
            parts = []
            offset = 0
            for samples in all_samples:
                parts.append(samples[node].indices[column] + offset)

                if column == "dataset":
                    offset += num_datasets(samples)
                else:
                    offset += num_rows(samples[column].indices)

            indices[column] = np.concatenate(parts)

        values = {}

        for variable in first.values:
            parts = []
            for samples in all_samples:
                parts.append(samples[node].values[variable])

            values[variable] = np.concatenate(parts)

        concatenated[node] = NodeSamples(indices, values)

    return concatenated


def select(samples: GraphSamples, datasets: np.ndarray) -> GraphSamples:
    """
    Keeps only some datasets of the samples.

    The selected datasets are renumbered from 0 in increasing order of their
    original index, and the index columns are renumbered to match.

    Parameters
    ----------
    samples : GraphSamples
        The samples of every node.
    datasets : np.ndarray
        The numbers of the datasets to keep.

    Returns
    -------
    GraphSamples
        Samples holding only the selected datasets.
    """
    # select only indices and values with a specific dataset ID

    datasets = np.sort(datasets)

    rows = {"dataset": datasets}
    for node, node_samples in samples.items():
        starts = np.searchsorted(node_samples.indices["dataset"], datasets, side="left")
        ends = np.searchsorted(node_samples.indices["dataset"], datasets, side="right")
        rows[node] = concatenated_ranges(starts, ends - starts)

    selected = {}
    for node, node_samples in samples.items():
        indices = {}
        for column, ids in node_samples.indices.items():
            # renumbers ids by their position among the selected rows
            indices[column] = np.searchsorted(rows[column], ids[rows[node]])

        values = {}
        for variable, value in node_samples.values.items():
            values[variable] = value[rows[node]]

        selected[node] = NodeSamples(indices, values)

    return selected
