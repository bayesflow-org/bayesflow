from typing import Literal

import keras

from bayesflow.types import Tensor

from .tensor_concatenation import concatenate


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
