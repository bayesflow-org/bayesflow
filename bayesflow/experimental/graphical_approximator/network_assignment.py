from typing import TYPE_CHECKING

import keras
import sympy as sp

from bayesflow.experimental.graphs import approximator_helpers
from bayesflow.experimental.graphs.utils import sort_nodes_topologically

from .shape_inference import inference_variable_shapes_by_network, summary_output_shapes_by_network
from .tensor_concatenation import concatenate

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator
from bayesflow.types import Tensor


def inference_variables_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference variables of that inference network.
    """
    result = {}

    for network_idx, _ in enumerate(approximator.inference_networks):
        result[network_idx] = _prepare_inference_variables(approximator, adapted_data, network_idx)

    return result


def _prepare_inference_variables(approximator: "GraphicalApproximator", adapted_data: dict, network_idx: int) -> Tensor:
    """
    Returns the inference variable tensor for the inference network denoted by `network_idx`.
    """
    vars = []
    for node in approximator.network_composition[network_idx]:
        for name in approximator.variable_names[node]:
            var = adapted_data[name]

            # standardize inference variables if required
            if name in approximator.standardize:
                var = approximator.standardize_layers[name](var, stage="validation")

            vars.append(var)

    inference_variables = concatenate(vars)

    return inference_variables


def summary_input(approximator: "GraphicalApproximator", adapted_data: dict) -> Tensor:
    """
    Returns the input for the first summary network.
    """
    data_node = approximator.graph.simulation_graph.data_node()
    data_variables = approximator.graph.simulation_graph.variable_names()[data_node]

    shapes = []

    for variable in data_variables:
        shapes.append(adapted_data[variable])

    return concatenate(shapes)


def summary_outputs_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are the outputs of that summary network.
    """
    input_tensor = summary_input(approximator, adapted_data)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        output_tensor = summary_network(input_tensor, training=False)
        result[i] = output_tensor

        input_tensor = output_tensor

    return result


def summary_inputs_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are the inputs of that summary network.
    """
    input_tensor = summary_input(approximator, adapted_data)
    input_shape = keras.ops.shape(input_tensor)

    result = {}

    # regular summary networks
    for i, summary_network in enumerate(approximator.summary_networks or []):
        result[i] = input_tensor
        output_tensor = summary_network(input_tensor, training=False)
        # next summary network uses previous output as input
        input_tensor = output_tensor

    # extra summary networks required for reshaped inputs
    input_tensor = summary_input(approximator, adapted_data)
    inference_variables = inference_variables_by_network(approximator, adapted_data)
    variable_shapes = {k: keras.ops.shape(v) for k, v in inference_variables.items()}

    network_idx = max(result.keys()) + 1
    for i, variable_shape in variable_shapes.items():
        if variable_shape[:-1] != input_shape[: len(variable_shape[:-1])]:
            prefix = variable_shape[:-1]

            reshaped_input = _permute_to_prefix(input_tensor, prefix)
            result[network_idx] = reshaped_input

            for i in range(len(prefix), len(reshaped_input) - 2):
                print(i)
                network_idx += 1
                result[network_idx] = approximator.summary_networks[network_idx - 1](result[network_idx - 1])

    return result


def data_conditions_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """ ""
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data conditions of that inference network.
    """
    inference_variables = inference_variables_by_network(approximator, adapted_data)

    result = {}

    for network_idx, variable in inference_variables.items():
        result[network_idx] = _prepare_data_conditions(approximator, adapted_data, network_idx)

    return result


def _prepare_data_conditions(
    approximator: "GraphicalApproximator", adapted_data: dict, network_idx: int, meta_dict: dict | None = None
) -> Tensor:
    """
    Returns the data condition tensor for the inference network denoted by `network_idx`.
    """
    if meta_dict is not None:
        # we pass meta_dict only if sampling from approximator
        inference_variable_shapes = inference_variable_shapes_by_network(approximator, meta_dict=meta_dict)
        inference_var_rank = len(inference_variable_shapes[network_idx])
        inference_group_dim = inference_variable_shapes[network_idx][-2]
    else:
        inference_vars = _prepare_inference_variables(approximator, adapted_data, network_idx)
        inference_var_rank = keras.ops.ndim(inference_vars)
        inference_group_dim = keras.ops.shape(inference_vars)[-2]

    summary_outputs = summary_outputs_by_network(approximator, adapted_data)
    inference_to_summary_map = match_inference_to_summary_networks(approximator)

    summary_idx = inference_to_summary_map[network_idx]
    summary_output = summary_outputs[summary_idx]

    if inference_var_rank == len(summary_output.shape):
        data_conditions = summary_output
    else:
        expanded = keras.ops.expand_dims(summary_output, axis=-2)
        broadcasted = keras.ops.broadcast_to(
            expanded,
            (
                *keras.ops.shape(summary_output)[:-1],
                inference_group_dim,
                keras.ops.shape(summary_output)[-1],
            ),
        )
        data_conditions = broadcasted

    return data_conditions


def match_inference_to_summary_networks(approximator: "GraphicalApproximator"):
    variable_shapes = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)
    condition_shapes = approximator_helpers.inference_condition_shapes_by_network(approximator.graph)
    summary_inputs = approximator_helpers.summary_input_shapes_by_network(approximator.graph)

    result = {}

    for k, v in condition_shapes.items():
        if isinstance(v[-1], sp.Basic):
            symbols = list(v[-1].free_symbols)
            for symbol in symbols:
                idx = int(symbol.name.split("_")[-1])
                if summary_inputs[k] not in variable_shapes.values():
                    result[k] = idx

    return result


def inference_conditions_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference variables of that inference network.
    """
    result = {}

    for network_idx, _ in enumerate(approximator.inference_networks):
        result[network_idx] = _prepare_inference_conditions(approximator, adapted_data, network_idx)
    return result


def _prepare_inference_conditions(
    approximator: "GraphicalApproximator", adapted_data: dict, network_idx: int, meta_dict: dict | None = None
) -> Tensor:
    """
    Returns the inference condition tensor for the inference network denoted by `network_idx`.
    """
    data_node = approximator.graph.simulation_graph.data_node()
    data_conditions = _prepare_data_conditions(approximator, adapted_data, network_idx, meta_dict=meta_dict)
    variable_names = approximator.variable_names

    vars = []

    for node in sort_nodes_topologically(
        approximator.graph.simulation_graph, approximator.network_conditions[network_idx]
    ):
        if node in data_node:
            vars.append(data_conditions)
        else:
            for name in variable_names[node]:
                var = adapted_data[name]

                # standardize inference variables if required
                if name in approximator.standardize:
                    var = approximator.standardize_layers[name](var, stage="validation")

                # flatten group dimension if node is not amortizable
                if not approximator.graph.allows_amortization(node):
                    # transpose last two dimensions before flattening
                    # so unpacking in split_network_output becomes easier
                    rank = keras.ops.ndim(var)
                    perm = (*range(rank - 2), rank - 1, rank - 2)
                    transpose = keras.ops.transpose(var, axes=tuple(perm))
                    var = keras.ops.reshape(transpose, (*keras.ops.shape(transpose)[:-2], -1))

                vars.append(var)

    inference_conditions = concatenate(vars)

    # add node repetitions
    input = summary_input(approximator, adapted_data)
    node_reps = keras.ops.shape(input)[1:-1]
    if len(node_reps) >= 1:
        squared = keras.ops.sqrt(node_reps)
        expanded = keras.ops.expand_dims(squared, axis=0)
        repeated = keras.ops.repeat(expanded, keras.ops.shape(input)[0], axis=0)
        inference_conditions = concatenate([inference_conditions, repeated])

    return inference_conditions


def _permute_to_prefix(source_tensor, prefix_shape):
    """
    Mutates the source shape in such a way that it starts with `prefix`.
    """
    print(":")
    print(keras.ops.shape(source_tensor))
    print(prefix_shape)
    source_shape = keras.ops.shape(source_tensor)
    prefix_indices = [list(source_shape).index(s) for s in prefix_shape]
    remaining_indices = [i for i in range(len(source_shape)) if i not in prefix_indices]

    perm = prefix_indices + remaining_indices

    return keras.ops.transpose(source_tensor, perm)
