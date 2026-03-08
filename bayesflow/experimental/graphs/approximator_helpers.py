import sympy as sp

from ..graphical_approximator.shape_operations import concatenate_shapes
from .inverted_graph import InvertedGraph

"""
This module contains helper functions that are used to build the approximator,
for example, inferring the number of required inference and summary networks.
"""


def print_network_summary(graph: InvertedGraph):
    variable_shapes = inference_variable_shapes_by_network(graph)
    network_composition = graph.network_composition()
    summary_inputs = summary_input_shapes_by_network(graph)
    summary_outputs = summary_output_shapes_by_network(graph)

    print(f"The approximator requires {len(variable_shapes)} inference networks:")
    for k, v in variable_shapes.items():
        print(f"  inference network {k}:")
        print(f"    shape {v} for nodes {network_composition[k]})")

    print("")

    print(f"The approximator requires {len(summary_inputs)} summary networks:")
    for k, v in summary_inputs.items():
        print(f"  summary network {k}:")
        print(f"    {v} -> {summary_outputs[k]}")


def inference_variable_shapes_by_network(graph: InvertedGraph) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that inference network.
    """
    output_shapes = graph.simulation_graph.output_shapes()
    network_composition = graph.network_composition()
    variable_names = graph.simulation_graph.variable_names()

    result = {}

    for i, _ in enumerate(network_composition):
        variable_shapes = []

        for node in network_composition[i]:
            for variable in variable_names[node]:
                if variable in output_shapes:
                    variable_shapes.append(output_shapes[variable])

        result[i] = concatenate_shapes(variable_shapes)

    return result


def summary_input_shapes_by_network(graph: InvertedGraph) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are input shapes of that summary network.
    """

    variable_shapes = inference_variable_shapes_by_network(graph)
    input_shape = _summary_input_shape(graph)
    result = {0: input_shape}

    # regular summary networks from summary_input_shape -> (B, summary_input)
    for i in range(1, len(input_shape) - 2):
        result[i] = result[i - 1][:-2] + (sp.Symbol(f"summary_dim_{i - 1}"),)

    # extra summary networks required for reshaped inputs
    network_idx = max(result.keys()) + 1
    for i, variable_shape in variable_shapes.items():
        if variable_shape[:-1] != input_shape[: len(variable_shape[:-1])]:
            prefix = variable_shape[:-1]
            reshaped_input = _permute_to_prefix(input_shape, prefix)

            result[network_idx] = reshaped_input
            for j in range(len(prefix), len(reshaped_input) - 2):
                result[network_idx] = result[network_idx - 1][:-2] + (sp.Symbol(f"summary_dim_{network_idx}"),)
                network_idx += 1

    # extra summary networks required for non-exchangeable nodes
    network_composition = graph.network_composition()

    network_idx = max(result.keys()) + 1
    for i, composition in network_composition.items():
        for node in composition:
            if not graph.allows_amortization(node):
                result[network_idx] = variable_shapes[i]
                network_idx += 1
                continue

    return result


def summary_output_shapes_by_network(graph: InvertedGraph) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that summary network.
    """

    input_shapes = summary_input_shapes_by_network(graph)

    result = {}

    for i, input_shape in input_shapes.items():
        result[i] = input_shape[:-2] + (sp.Symbol(f"summary_dim_{i}"),)

    return result


def data_condition_shapes_by_network(graph: InvertedGraph) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data condition shapes of that inference network.
    """
    inference_variable_shapes = inference_variable_shapes_by_network(graph)
    summary_output_shapes = summary_output_shapes_by_network(graph)

    result = {}

    for network_idx, variable_shape in inference_variable_shapes.items():
        for k, v in summary_output_shapes.items():
            if variable_shape[:-1] == v[:-1]:
                result[network_idx] = v

        if network_idx not in result:
            for k, v in summary_output_shapes.items():
                try:
                    concatenate_shapes([variable_shape, v])
                    result[network_idx] = (variable_shape[:-1] + (v[-1],),)
                except Exception:
                    pass

    return result


def first_stage_condition_shapes_by_network(graph: InvertedGraph) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns the inference condition shapes for each network.
    This includes inference variables in the last dimension
    in the case of non-exchangeable nodes.
    """
    output_shapes = graph.simulation_graph.output_shapes()
    variable_names = graph.simulation_graph.variable_names()
    data_node = graph.simulation_graph.data_node()

    data_condition_shapes = data_condition_shapes_by_network(graph)
    network_conditions = graph.network_conditions()
    summary_input = _summary_input_shape(graph)

    result = {}

    for i, _ in data_condition_shapes.items():
        condition_shapes = []
        for node in set(network_conditions[i]):
            if node in data_node:
                condition_shapes.append(data_condition_shapes[i])
            else:
                for variable in variable_names[node]:
                    shape = output_shapes[variable]

                    # flatten group dimension if node is not amortizable
                    if not graph.allows_amortization(node):
                        shape = shape[:-2] + (sp.prod(shape[-2:]),)

                    condition_shapes.append(shape)

        result[i] = list(concatenate_shapes(condition_shapes))
        result[i][-1] += len(summary_input[1:-1])  # add node repetitions
        result[i] = tuple(x for x in result[i])

    return result


def _permute_to_prefix(source_shape, prefix):
    """
    Mutates the source shape in such a way that it starts with `prefix`.
    """
    prefix_indices = [list(source_shape).index(s) for s in prefix]
    remaining_indices = [i for i in range(len(source_shape)) if i not in prefix_indices]

    perm = prefix_indices + remaining_indices

    return tuple(source_shape[i] for i in perm)


def _summary_input_shape(graph: InvertedGraph) -> tuple[int | sp.Expr, ...]:
    """
    Returns the input for the first summary network.
    """
    data_node = graph.simulation_graph.data_node()
    data_variables = graph.simulation_graph.variable_names()[data_node]
    output_shapes = graph.simulation_graph.output_shapes()

    shapes = []

    for variable in data_variables:
        shapes.append(output_shapes[variable])

    return concatenate_shapes(shapes)
