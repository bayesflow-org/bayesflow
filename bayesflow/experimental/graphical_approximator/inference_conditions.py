from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import keras

from bayesflow.experimental.graphs.inverted_graph import SummaryKey
from bayesflow.networks import SummaryNetwork
from bayesflow.types import Tensor

from .tensor_concatenation import concatenate

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator


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

    # only permute dimensions that are present in source; missing prefix dims
    # are handled later by expand_to_prefix
    shared_prefix = [dim for dim in target_prefix if dim in source]
    prefix_indices = [source.index(dim) for dim in shared_prefix]
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
        Input tensor to summarize.
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


def compute_chain_steps(
    tensor: Tensor,
    shared_prefix: tuple,
    mode: Literal["global", "per_level"],
) -> int:
    """
    Returns the number of summary networks to apply in the parallel chain.

    Parameters
    ----------
    tensor : Tensor
        The permuted conditioned tensor.
    shared_prefix : tuple
        The shared prefix dimensions between the conditioned and inferred nodes.
    mode : {"global", "per_level"}
        Summary mode.

    Returns
    -------
    int
        Number of chain steps. Zero means no summary network is needed.
    """
    n_extra = tensor.ndim - 1 - len(shared_prefix)
    if mode == "per_level":
        n_extra -= 1
    return max(0, n_extra)


def apply_parallel_chain(
    tensor: Tensor,
    conditioned_node: str,
    mode: Literal["global", "per_level"],
    inferred_node: str | None,
    k: int,
    registry: dict[SummaryKey, SummaryNetwork],
    training: bool = False,
) -> Tensor:
    """
    Applies k summary networks sequentially from innermost to outermost
    extra dimension, maintaining a carry for enrichment at each step.

    Parameters
    ----------
    tensor : Tensor
        The permuted conditioned tensor.
    conditioned_node : str
        The node being summarized.
    mode : {"global", "per_level"}
        Summary mode.
    inferred_node : str or None
        The inferred node, required for per_level mode.
    k : int
        Number of chain steps, as returned by compute_chain_steps.
    registry : dict[SummaryKey, SummaryNetwork]
        Mapping from summary keys to summary networks.
    training : bool, optional
        Whether the model is in training mode. Default is False.

    Returns
    -------
    Tensor
        Output of the final summary network in the chain.
    """
    carry = keras.ops.mean(tensor, axis=-2)
    summary = None

    for i in range(k):
        if i == 0:
            chain_input = tensor
        else:
            chain_input = concatenate([carry, summary])
            carry = keras.ops.mean(summary, axis=-2)

        key = SummaryKey(
            conditioned_node=conditioned_node,
            mode=mode,
            inferred_node=inferred_node if mode == "per_level" else None,
            chain_step=i,
        )
        summary = apply_summary(chain_input, key, registry, training=training)

    return summary


def inference_conditions_by_network(
    approximator: GraphicalApproximator,
    simulation_output: dict[str, Tensor],
    summary_registry: dict[SummaryKey, SummaryNetwork],
    training: bool = False,
    only_network: int | None = None,
) -> dict[int, Tensor]:
    """
    Computes inference conditions for each network by running each conditioned
    node's output through the full condition pipeline: gather -> permute ->
    compute_chain_steps -> apply_parallel_chain -> expand.

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
    only_network : int or None, optional
        If given, only compute conditions for this network index. Used during
        sequential sampling where later networks' conditions are not yet available.

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
        if only_network is not None and network_idx != only_network:
            continue
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
            shared_prefix = [dim for dim in inferred_prefix if dim in source_shape]
            tensor = permute_to_prefix(tensor, source_shape, inferred_prefix)

            if tensor.ndim > len(shared_prefix) + 1:
                if approximator.graph.is_per_level_summary(inferred_nodes[0], conditioned_node):
                    mode = "per_level"
                else:
                    mode = "global"
                k = compute_chain_steps(tensor, shared_prefix, mode)
                tensor = apply_parallel_chain(
                    tensor,
                    conditioned_node=conditioned_node,
                    mode=mode,
                    inferred_node=inferred_nodes[0] if mode == "per_level" else None,
                    k=k,
                    registry=summary_registry,
                    training=training,
                )

            # from_prefix is always recoverable from the tensor rank after the
            # pipeline steps above, since permute aligns dimensions with the
            # inferred prefix
            from_prefix = inferred_prefix[: tensor.ndim - 1]
            tensor = expand_to_prefix(tensor, from_prefix, inferred_prefix)
            condition_tensors.append(tensor)

        result[network_idx] = concatenate(condition_tensors)

    return result
