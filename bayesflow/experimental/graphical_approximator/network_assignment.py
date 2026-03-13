from typing import TYPE_CHECKING

import keras
import sympy as sp

from bayesflow.experimental.graphs import approximator_helpers
from bayesflow.experimental.graphs.utils import sort_nodes_topologically

from .shape_operations import concatenate_shapes
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
        try:
            result[network_idx] = _prepare_inference_variables(approximator, adapted_data, network_idx)
        except KeyError:
            pass

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
    summary_inputs = summary_inputs_by_network(approximator, adapted_data)

    result = {}

    for i, input_tensor in summary_inputs.items():
        output_tensor = approximator.summary_networks[i](input_tensor, training=False)
        result[i] = output_tensor

    return result


def summary_inputs_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are the inputs of that summary network.
    """
    input_tensor = summary_input(approximator, adapted_data)

    result = {}

    # regular summary networks
    for i, summary_network in enumerate(approximator.summary_networks or []):
        result[i] = input_tensor
        output_tensor = summary_network(input_tensor, training=False)
        # next summary network uses previous output as input
        input_tensor = output_tensor

        if input_tensor.ndim == 2:
            break

    # extra summary networks required for reshaped inputs
    input_tensor = summary_input(approximator, adapted_data)
    symbolic_var_shapes = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)
    symbolic_input_shape = approximator_helpers.summary_input_shapes_by_network(approximator.graph)[0]

    network_idx = max(result.keys()) + 1
    for i, symbolic_var_shape in symbolic_var_shapes.items():
        if symbolic_var_shape[:-1] != symbolic_input_shape[: len(symbolic_var_shape[:-1])]:
            src = list(symbolic_input_shape)
            prefix = list(symbolic_var_shape[:-1])
            prefix_indices = [src.index(s) for s in prefix]
            remaining_indices = [j for j in range(len(src)) if j not in prefix_indices]
            perm = prefix_indices + remaining_indices

            reshaped_input = keras.ops.transpose(input_tensor, perm)
            result[network_idx] = reshaped_input

            for _ in range(len(prefix), len(reshaped_input.shape) - 2):
                network_idx += 1
                result[network_idx] = approximator.summary_networks[network_idx - 1](result[network_idx - 1])

    # extra summary networks required for non-exchangeable nodes
    network_composition = approximator.network_composition
    inference_variables = None

    network_idx = max(result.keys()) + 1
    for i, composition in network_composition.items():
        for node in composition:
            if not approximator.graph.allows_amortization(node):
                if inference_variables is None:
                    inference_variables = inference_variables_by_network(approximator, adapted_data)
                if i in inference_variables:
                    result[network_idx] = inference_variables[i]
                    network_idx += 1
                continue

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
    variable_shapes = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)[network_idx]
    summary_output_shapes = approximator_helpers.summary_output_shapes_by_network(approximator.graph)
    summary_outputs = summary_outputs_by_network(approximator, adapted_data)

    result = None

    for i, v in summary_output_shapes.items():
        if variable_shapes[:-1] == v[:-1]:
            result = summary_outputs[i]
            break

    if result is None:
        for i, v in summary_output_shapes.items():
            try:
                concatenate_shapes([variable_shapes, v])
                result = summary_outputs[i]
            except Exception:
                pass

    if result is None:
        raise RuntimeError(f"No matching summary output found for inference network with index {network_idx}.")

    return result


def match_inference_to_summary_networks(approximator: "GraphicalApproximator"):
    variable_shapes = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)
    condition_shapes = approximator_helpers.inference_condition_shapes_by_network(approximator.graph)
    summary_inputs = approximator_helpers.summary_input_shapes_by_network(approximator.graph)

    result = {}

    for k, v in condition_shapes.items():
        last = v[-1]
        if isinstance(last, sp.Expr):
            symbols = list(last.free_symbols)
            for symbol in symbols:
                idx = int(str(symbol).split("_")[-1])
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

                if approximator.graph.allows_amortization(node):
                    vars.append(var)

    # add non-exchangeable variables as conditions after summarizing
    symbolic_expr = approximator_helpers.inference_condition_shapes_by_network(approximator.graph)[network_idx][-1]

    if not isinstance(symbolic_expr, int):
        inference_variable_shapes = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)
        summary_input_shapes = approximator_helpers.summary_input_shapes_by_network(approximator.graph)
        summary_outputs = summary_outputs_by_network(approximator, adapted_data)

        for k in range(len(approximator.summary_networks)):
            if sp.Symbol(f"summary_dim_{k}") in symbolic_expr.free_symbols:
                if summary_input_shapes[k] in inference_variable_shapes.values():
                    vars.append(summary_outputs[k])

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
    source_shape = source_tensor.shape
    prefix_indices = [list(source_shape).index(s) for s in prefix_shape]
    remaining_indices = [i for i in range(len(source_shape)) if i not in prefix_indices]

    perm = prefix_indices + remaining_indices

    return keras.ops.transpose(source_tensor, perm)
