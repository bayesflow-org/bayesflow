from typing import TYPE_CHECKING

import keras
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
    """ ""
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data conditions of that inference network.
    """
    inference_variables = inference_variables_by_network(approximator, adapted_data)
    summary_outputs = summary_outputs_by_network(approximator, adapted_data)
    inference_to_summary_map = match_inference_to_summary_networks(approximator)

    result = {}

    for network_idx, variable in inference_variables.items():
        summary_idx = inference_to_summary_map[network_idx]
        summary_output = summary_outputs[summary_idx]

        if len(variable.shape) == len(summary_output.shape):
            result[network_idx] = summary_output
        else:
            expanded = keras.ops.expand_dims(summary_output, axis=-2)
            broadcasted = keras.ops.broadcast_to(
                expanded,
                (
                    *keras.ops.shape(summary_output)[:-1],
                    keras.ops.shape(variable)[-2],
                    keras.ops.shape(summary_output)[-1],
                ),
            )
            result[network_idx] = broadcasted

    return result


def match_inference_to_summary_networks(approximator: "GraphicalApproximator") -> dict[int, int]:
    """
    Match each inference network to a summary network that outputs the necessary data conditions.
    """

    def num_equal_dims(a, b):
        i = 0
        for x, y in zip(a, b):
            if x == y:
                i += 1
            else:
                break

        return i

    inference_variable_shapes = inference_variable_shapes_by_network(approximator)
    summary_output_shapes = summary_output_shapes_by_network(approximator)

    summary_output_by_rank = {len(v): v for _, v in summary_output_shapes.items()}
    network_idx_by_rank = {len(v): k for k, v in summary_output_shapes.items()}

    inference_to_summary_map = {}

    for k, v in inference_variable_shapes.items():
        rank = len(v)
        summary_output = summary_output_by_rank[rank]

        if num_equal_dims(summary_output, v) == rank - 1:
            inference_to_summary_map[k] = network_idx_by_rank[rank]
        else:
            inference_to_summary_map[k] = network_idx_by_rank[rank - 1]

    return inference_to_summary_map


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

        for node in sort_nodes_topologically(
            approximator.graph.simulation_graph, approximator.network_conditions[network_idx]
        ):
            if node in data_node:
                vars.append(data_conditions[network_idx])
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

        conditions = concatenate(vars)

        # add node repetitions
        input = summary_input(approximator, adapted_data)
        node_reps = keras.ops.shape(input)[1:-1]
        if len(node_reps) >= 1:
            squared = keras.ops.sqrt(node_reps)
            expanded = keras.ops.expand_dims(squared, axis=0)
            repeated = keras.ops.repeat(expanded, keras.ops.shape(input)[0], axis=0)
            conditions = concatenate([conditions, repeated])

        result[network_idx] = conditions

    return result
