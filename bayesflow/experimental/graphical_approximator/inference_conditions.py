from typing import TYPE_CHECKING, Literal

import keras
import sympy as sp

from bayesflow.experimental.graphs.inverted_graph import SummaryKey
from bayesflow.networks import SummaryNetwork
from bayesflow.types import Tensor

from .inference_variables import resolve_shapes
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

    Parameters
    ----------
    tensor : Tensor
        The tensor to permute.
    source_shape : tuple
        Full symbolic shape of `tensor`, e.g. ``(B, N_globals, N_locals, D)``.
        Typically obtained from ``SimulationGraph.output_shapes()``.
    target_prefix : tuple
        The desired leading dimensions, e.g. ``(B, N_locals)``.
        Every element must appear in `source_shape`.

    Returns
    -------
    Tensor
        Tensor with axes reordered so that `target_prefix` dimensions come first.
    """
    source = list(source_shape)

    shared_prefix = [dim for dim in target_prefix if dim in source]
    prefix_indices = [source.index(dim) for dim in shared_prefix]
    remaining_indices = [i for i in range(len(source)) if i not in prefix_indices]
    perm = prefix_indices + remaining_indices

    # return original tensor if no transpose is necessary
    if perm == list(range(len(source))):
        return tensor

    return keras.ops.transpose(tensor, perm)


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
        Target prefix to expand towards, e.g. ``(B, N_globals, N_locals)``.
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
    network = registry[key]

    # flattening and reshaping are done because DeepSets do not inputs with more than 3 dimensions yes
    if tensor.ndim > 3:
        shape = keras.ops.shape(tensor)
        tensor = keras.ops.reshape(tensor, (-1, shape[-2], shape[-1]))
        out = network(tensor, training=training)

        return keras.ops.reshape(out, (*shape[:-2], out.shape[-1]))

    return network(tensor, training=training)


def compute_chain_steps(
    tensor: Tensor,
    shared_prefix: tuple,
) -> int:
    """
    Returns the number of summary networks to apply.

    Parameters
    ----------
    tensor : Tensor
        The permuted conditioned tensor.
    shared_prefix : tuple
        The shared prefix dimensions between the conditioned and inferred nodes.

    Returns
    -------
    int
        Number of chain steps. Zero means no summary network is needed.
    """
    return max(0, tensor.ndim - 1 - len(shared_prefix))


def apply_sequential_chain(
    tensor: Tensor,
    conditioned_node: str,
    mode: Literal["global", "per_level"],
    inferred_nodes: tuple[str, ...],
    k: int,
    registry: dict[SummaryKey, SummaryNetwork],
    training: bool = False,
) -> Tensor:
    """
    Applies k summary networks sequentially from innermost to outermost
    extra dimension, maintaining the previous mean at each step.

    Parameters
    ----------
    tensor : Tensor
        The permuted conditioned tensor.
    conditioned_node : str
        The node being summarized.
    mode : {"global", "per_level"}
        Summary mode.
    inferred_nodes : tuple of str
        For ``"per_level"`` mode, a 1-tuple of the inferred node that determines
        which summary network to use. For ``"global"`` mode, a tuple of all
        inferred nodes that share this summary.
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
            inferred_nodes=inferred_nodes,
            chain_step=i,
        )
        summary = apply_summary(chain_input, key, registry, training=training)

    return summary


def inference_conditions_by_network(
    approximator: "GraphicalApproximator",
    simulation_output: dict[str, Tensor],
    summary_registry: dict[SummaryKey, SummaryNetwork],
    training: bool = False,
    only_network: int | None = None,
) -> dict[int, Tensor]:
    """
    Computes inference conditions for each network by running each conditioned
    node's output through the full condition pipeline: gather -> permute ->
    compute_chain_steps -> apply_sequential_chain -> expand.

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

    # sqrt of the data node's spatial dimensions, appended to every network's conditions
    data_node = approximator.graph.simulation_graph.data_node()
    data_tensor = gather_node_output(variable_names[data_node], simulation_output)
    data_shape = keras.ops.shape(data_tensor)
    spatial_dims = [data_shape[i] for i in range(1, data_tensor.ndim - 1)]
    node_reps = keras.ops.expand_dims(
        keras.ops.sqrt(keras.ops.cast(keras.ops.stack(spatial_dims), "float32")),
        axis=0,
    )

    global_inferred = approximator.graph._global_summary_inferred_nodes()

    for network_idx, inferred_nodes in approximator.network_composition.items():
        if only_network is not None and network_idx != only_network:
            continue

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
                k = compute_chain_steps(tensor, shared_prefix)
                tensor = apply_sequential_chain(
                    tensor,
                    conditioned_node=conditioned_node,
                    mode=mode,
                    inferred_nodes=(inferred_nodes[0],) if mode == "per_level" else global_inferred[conditioned_node],
                    k=k,
                    registry=summary_registry,
                    training=training,
                )

            from_prefix = inferred_prefix[: tensor.ndim - 1]
            tensor = expand_to_prefix(tensor, from_prefix, inferred_prefix)
            condition_tensors.append(tensor)

        condition_tensors.append(node_reps)
        result[network_idx] = concatenate(condition_tensors)

    return result


def inference_condition_shapes_by_network(
    approximator: "GraphicalApproximator",
    data_shapes: dict | None = None,
    meta_dict: dict | None = None,
) -> dict[int, tuple[int | sp.Expr, ...]]:
    if not data_shapes:
        data_shapes = approximator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes)
    output_shapes = resolve_shapes(approximator.output_shapes, meta_dict)
    variable_names = approximator.variable_names

    data_node = approximator.graph.simulation_graph.data_node()
    n_node_reps = len(output_shapes[variable_names[data_node][0]]) - 2

    global_inferred = approximator.graph._global_summary_inferred_nodes()
    result = {}

    for network_idx, inferred_nodes in approximator.network_composition.items():
        inferred_prefix = output_shapes[variable_names[inferred_nodes[0]][0]][:-1]

        total_feature_dim = 0

        for conditioned_node in approximator.network_conditions[network_idx]:
            conditioned_vars = variable_names[conditioned_node]
            total_D = sum(output_shapes[v][-1] for v in conditioned_vars)
            source_shape = output_shapes[conditioned_vars[0]]

            shared_prefix = [dim for dim in inferred_prefix if dim in source_shape]
            k = len(source_shape) - 1 - len(shared_prefix)

            if k <= 0:
                total_feature_dim += total_D
            else:
                mode = (
                    "per_level"
                    if approximator.graph.is_per_level_summary(inferred_nodes[0], conditioned_node)
                    else "global"
                )
                key = SummaryKey(
                    conditioned_node=conditioned_node,
                    mode=mode,
                    inferred_nodes=(inferred_nodes[0],) if mode == "per_level" else global_inferred[conditioned_node],
                    chain_step=k - 1,
                )
                total_feature_dim += approximator.summary_registry[key].summary_dim

        total_feature_dim += n_node_reps
        result[network_idx] = inferred_prefix + (total_feature_dim,)

    return result
