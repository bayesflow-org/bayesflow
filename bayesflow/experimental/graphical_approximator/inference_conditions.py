from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal, NamedTuple

import keras

from bayesflow.networks import SummaryNetwork
from bayesflow.types import Tensor

from .tensor_concatenation import concatenate

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator


class SummaryKey(NamedTuple):
    """
    Identifies a summary network in the registry.

    For ``"global"`` mode, ``inferred_node`` is ``None`` because the same
    summary (over all data) is shared across all inferred nodes. For
    ``"per_level"`` mode, ``inferred_node`` identifies which level is kept
    non-flattened, so different inferred nodes at different levels get
    different summary networks.
    """

    conditioned_node: str
    mode: Literal["global", "per_level"]
    inferred_node: str | None = None


def gather_node_output(
    variable_names: list[str],
    simulation_output: dict[str, Tensor],
) -> Tensor:
    """
    Collects all output tensors for a node from `simulation_output` and
    concatenates them along the last axis.

    Parameters
    ----------
    variable_names : list[str]
        The variable names produced by the node, typically one entry of
        ``SimulationGraph.variable_names()``.
    simulation_output : dict[str, Tensor]
        Dictionary mapping variable names to tensors, as produced by the simulator.

    Returns
    -------
    Tensor
        Concatenation of all tensors along the last axis.
    """
    tensors = [simulation_output[name] for name in variable_names]
    return concatenate(tensors)


def permute_to_prefix(
    tensor: Tensor,
    source_shape: tuple,
    target_prefix: tuple,
) -> Tensor:
    """
    Reorders the axes of `tensor` so that the dimensions in `target_prefix`
    appear first, in that order. The remaining dimensions follow in their
    original relative order.

    The permutation is computed from symbolic shapes rather than the runtime
    tensor shape.

    Parameters
    ----------
    tensor : Tensor
        The tensor to permute.
    source_shape : tuple
        Full symbolic shape of `tensor`, e.g. ``(B, N_regions, N_squares, D)``.
        Typically obtained from ``SimulationGraph.output_shapes()``.
    target_prefix : tuple
        The desired leading dimensions, e.g. ``(B, N_squares)``.
        Every element must appear in `source_shape`.

    Returns
    -------
    Tensor
        Tensor with axes reordered so that `target_prefix` dimensions come first.
    """
    source = list(source_shape)
    prefix_indices = [source.index(dim) for dim in target_prefix]
    remaining_indices = [i for i in range(len(source)) if i not in prefix_indices]
    perm = prefix_indices + remaining_indices

    # return original tensor if no transpose is necessary
    if perm == list(range(len(source))):
        return tensor

    return keras.ops.transpose(tensor, perm)


def flatten_to_summary_input(
    tensor: Tensor,
    mode: Literal["global", "per_level"],
) -> Tensor:
    """
    Flattens `tensor` into the shape expected by a summary network, based on
    the summary mode. Assumes `permute_to_prefix` has already been applied so
    that the target prefix dimensions are leading.

    For ``"global"`` mode, all spatial dimensions are collapsed into a single
    set dimension: ``(B, d1, ..., dk, D) -> (B, d1*...*dk, D)``.

    For ``"per_level"`` mode, the first spatial dimension is kept and the
    remaining ones are collapsed: ``(B, N, d2, ..., dk, D) -> (B, N, d2*...*dk, D)``.

    Parameters
    ----------
    tensor : Tensor
        Input tensor. Must have rank >= 3 for ``"global"`` and >= 4 for
        ``"per_level"``.
    mode : {"global", "per_level"}
        Summary mode. ``"global"`` produces a single summary vector per batch
        element. ``"per_level"`` produces one summary per entry in the level
        dimension.

    Returns
    -------
    Tensor
        Tensor with spatial dimensions collapsed into a single set dimension.
    """
    shape = keras.ops.shape(tensor)

    if mode == "global":
        if tensor.ndim <= 3:
            return tensor
        return keras.ops.reshape(tensor, [shape[0], -1, shape[-1]])

    # per_level: keep the first spatial dimension, flatten the rest
    if tensor.ndim <= 4:
        return tensor
    return keras.ops.reshape(tensor, [shape[0], shape[1], -1, shape[-1]])


def expand_to_prefix(
    tensor: Tensor,
    from_prefix: tuple,
    to_prefix: tuple,
) -> Tensor:
    """
    Inserts singleton dimensions so that `tensor` can broadcast to `to_prefix`.

    Each missing prefix dimension is inserted as a singleton axis immediately
    before the last (data) dimension. The actual tiling to the target size
    happens later when conditions are assembled via ``concatenate``.

    Parameters
    ----------
    tensor : Tensor
        Input tensor with prefix ``from_prefix``, i.e. shape
        ``(*from_prefix, D)``.
    from_prefix : tuple
        Current prefix of `tensor`, e.g. ``(B,)`` for a global summary.
    to_prefix : tuple
        Target prefix to expand towards, e.g. ``(B, N_regions, N_squares)``.
        Must be at least as long as `from_prefix`.

    Returns
    -------
    Tensor
        Tensor with singleton dimensions inserted, shape ``(*to_prefix, D)``
        where the new axes have size 1.
    """
    n_new_axes = len(to_prefix) - len(from_prefix)
    for _ in range(n_new_axes):
        tensor = keras.ops.expand_dims(tensor, axis=-2)
    return tensor


def apply_summary(
    tensor: Tensor,
    key: SummaryKey,
    registry: dict[SummaryKey, SummaryNetwork],
    training: bool = False,
) -> Tensor:
    """
    Applies the summary network identified by `key` to `tensor`.

    Parameters
    ----------
    tensor : Tensor
        Flattened input tensor, as produced by ``flatten_to_summary_input``.
    key : SummaryKey
        Identifies which summary network to use.
    registry : dict[SummaryKey, SummaryNetwork]
        Mapping from summary keys to summary networks, built from the ordered
        list passed to ``GraphicalApproximator``.
    training : bool, optional
        Whether the model is in training mode, affecting layers like dropout.
        Default is False.

    Returns
    -------
    Tensor
        Output of the summary network.
    """
    return registry[key](tensor, training=training)


def is_per_level_summary(
    inferred_node: str,
    conditioned_node: str,
    inverted_graph,
) -> bool:
    """
    Returns True if `conditioned_node` requires a per-level summary when
    conditioning `inferred_node`, False if a global summary is required.

    A per-level summary applies when, for every expanded copy of
    `inferred_node`, all expanded copies of `conditioned_node` in its
    conditions share the same last-digit suffix as the expanded inferred node.
    This means data can be split by level rather than summarised globally.

    Parameters
    ----------
    inferred_node : str
        Original simulation node name being inferred.
    conditioned_node : str
        Original simulation node name being conditioned on.
    inverted_graph : InvertedGraph
        The inverted graph providing the expanded condition structure.

    Returns
    -------
    bool
        True if a per-level summary is appropriate, False if global.
    """
    detailed = inverted_graph.detailed_conditions_by_node()
    original_names = inverted_graph.original_node_names()

    for expanded_inferred, conditions in detailed.items():
        if original_names[expanded_inferred] != inferred_node:
            continue

        expanded_conditioned = [c for c in conditions if original_names[c] == conditioned_node]
        if not expanded_conditioned:
            continue

        inferred_match = re.search(r"\d+$", expanded_inferred)
        if inferred_match is None:
            return False

        inferred_last_digit = inferred_match.group()[-1]
        for c in expanded_conditioned:
            c_match = re.search(r"\d+$", c)
            if c_match is None or c_match.group()[-1] != inferred_last_digit:
                return False

    return True


def inference_conditions_by_network(
    approximator: GraphicalApproximator,
    simulation_output: dict[str, Tensor],
    summary_registry: dict[SummaryKey, SummaryNetwork],
    training: bool = False,
) -> dict[int, Tensor]:
    """
    Computes inference conditions for each network by running each conditioned
    node's output through the full condition pipeline: gather → permute →
    flatten → summarise → expand.

    Parameters
    ----------
    approximator : GraphicalApproximator
        The approximator, providing graph structure, variable names, and
        output shapes.
    simulation_output : dict[str, Tensor]
        Dictionary mapping variable names to tensors, as produced by the
        simulator or accumulated during sampling.
    summary_registry : dict[SummaryKey, SummaryNetwork]
        Mapping from summary keys to summary networks, built from the list
        passed to ``GraphicalApproximator``.
    training : bool, optional
        Whether the model is in training mode, affecting layers like dropout.
        Default is False.

    Returns
    -------
    dict[int, Tensor]
        Mapping from network index to the concatenated conditions tensor for
        that network.
    """
    result = {}
    output_shapes = approximator.output_shapes
    variable_names = approximator.variable_names

    for network_idx, inferred_nodes in approximator.network_composition.items():
        # all nodes in a network share the same prefix; use the first variable
        # of the first node to read it (variables of a node share a prefix,
        # differing only in the last data dimension)
        inferred_node_vars = variable_names[inferred_nodes[0]]
        inferred_prefix = output_shapes[inferred_node_vars[0]][:-1]

        condition_tensors = []

        for conditioned_node in approximator.network_conditions[network_idx]:
            tensor = gather_node_output(variable_names[conditioned_node], simulation_output)

            # use the first variable's shape to compute the permutation; all
            # variables of a node share the same prefix
            source_shape = output_shapes[variable_names[conditioned_node][0]]
            tensor = permute_to_prefix(tensor, source_shape, inferred_prefix)

            if tensor.ndim > len(inferred_prefix) + 1:
                if is_per_level_summary(inferred_nodes[0], conditioned_node, approximator.graph):
                    mode = "per_level"
                else:
                    mode = "global"
                tensor = flatten_to_summary_input(tensor, mode)
                key = SummaryKey(
                    conditioned_node,
                    mode,
                    inferred_nodes[0] if mode == "per_level" else None,
                )
                tensor = apply_summary(tensor, key, summary_registry, training=training)

            # from_prefix is always recoverable from the tensor rank after the
            # pipeline steps above, since permute aligns dimensions with the
            # inferred prefix
            from_prefix = inferred_prefix[: tensor.ndim - 1]
            tensor = expand_to_prefix(tensor, from_prefix, inferred_prefix)
            condition_tensors.append(tensor)

        result[network_idx] = concatenate(condition_tensors)

    return result
