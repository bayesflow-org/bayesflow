from typing import TYPE_CHECKING

import keras

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
    variable_names = approximator.variable_names

    result = {}

    for network_idx, _ in enumerate(approximator.inference_networks):
        vars = []
        for node in approximator.network_composition[network_idx]:
            for name in variable_names[node]:
                var = adapted_data[name]

                # standardize inference variables if required
                if name in approximator.standardize:
                    var = approximator.standardize_layers[name](var, stage="validation")

                vars.append(var)

        result[network_idx] = concatenate(vars)

    return result


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

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        result[i] = input_tensor
        output_tensor = summary_network(input_tensor, training=False)
        # next summary network uses previous output as input
        input_tensor = output_tensor

    return result


def data_conditions_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data condition shapes of that inference network.
    """
    inference_variables = inference_variables_by_network(approximator, adapted_data)
    summary_outputs = summary_outputs_by_network(approximator, adapted_data)

    result = {}

    for network_idx, variable in inference_variables.items():
        for k, v in summary_outputs.items():
            if variable.shape[:-1] == v.shape[:-1]:
                result[network_idx] = v

        if network_idx not in result:
            for k, v in summary_outputs.items():
                try:
                    concatenate([variable, v])
                    expanded = keras.ops.expand_dims(v, axis=-2)
                    repeated = keras.ops.repeat(expanded, variable.shape[-2], axis=-2)
                    result[network_idx] = repeated
                except Exception:
                    pass

    return result


def inference_conditions_by_network(approximator: "GraphicalApproximator", adapted_data: dict) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference variables of that inference network.
    """
    variable_names = approximator.variable_names
    data_conditions = data_conditions_by_network(approximator, adapted_data)
    data_node = approximator.graph.simulation_graph.data_node()

    result = {}

    for network_idx, _ in enumerate(approximator.inference_networks):
        vars = []
        nodes_to_condition_on = (
            set(approximator.network_conditions) - {data_node} - set(approximator.network_composition)
        )

        for node in sort_nodes_topologically(approximator.graph.simulation_graph, list(nodes_to_condition_on)):
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

        vars.append(data_conditions[network_idx])
        conditions = concatenate(vars)

        # add node repetitions
        repetitions = node_repetitions(approximator, adapted_data)

        if repetitions != {}:
            conditions = add_node_reps_to_conditions(conditions, repetitions)

        result[network_idx] = conditions

    return result


def node_repetitions(approximator: "GraphicalApproximator", adapted_data: dict[str, int]) -> dict[str, int]:
    """
    Infers repetition counts for each node from data shapes.
    """
    data_shapes = approximator._data_shapes(adapted_data)
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    summary_input_shape = concatenate_shapes([data_shapes[k] for k in data_keys])
    shape_order = approximator.graph.data_shape_order()

    repetitions = {}

    for i, variable in enumerate(shape_order):
        repetitions[variable] = summary_input_shape[1:-1][i]  # skip batch and and variable dimension

    return repetitions


def add_node_reps_to_conditions(conditions, repetitions: dict[str, int]) -> Tensor:
    """
    Appends node repetition features (sqrt of node repetitons) to a conditions tensor.
    """
    rep_values = keras.ops.convert_to_tensor(list(repetitions.values()))
    squared = keras.ops.sqrt(rep_values)
    expanded = keras.ops.expand_dims(squared, axis=0)

    return concatenate([conditions, expanded])
