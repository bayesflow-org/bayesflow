from typing import Literal, NamedTuple

import keras

from bayesflow.networks import SummaryNetwork
from bayesflow.types import Tensor

from .tensor_concatenation import concatenate


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
