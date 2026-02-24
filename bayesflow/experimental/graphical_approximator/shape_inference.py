from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator

from .shape_operations import concatenate_shapes, replace_placeholders, resolve_shapes


def summary_input_shape(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> tuple[int | sp.Expr, ...]:
    """
    Returns the input for the first summary network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    output_shapes = resolve_shapes(data_shapes, meta_dict)
    data_node = approximator.graph.simulation_graph.data_node()
    data_variables = approximator.graph.simulation_graph.variable_names()[data_node]

    shapes = []

    for variable in data_variables:
        shapes.append(output_shapes[variable])

    return concatenate_shapes(shapes)


def summary_output_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that summary network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    input_shape = summary_input_shape(approximator, data_shapes, meta_dict)

    placeholders = [x for x in input_shape if isinstance(x, sp.Expr)]
    mock_meta = {k: 1 for k in placeholders} | {"B": 1}
    mock_input_shape = replace_placeholders(input_shape, mock_meta)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        output_shape = summary_network.compute_output_shape(mock_input_shape)
        result[i] = input_shape[: len(output_shape) - 1] + (output_shape[-1],)
        mock_input_shape = mock_input_shape[: len(output_shape) - 1] + (output_shape[-1],)

    return result


def summary_input_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are input shapes of that summary network.
    """

    if not data_shapes:
        data_shapes = approximator.output_shapes

    input_shape = summary_input_shape(approximator, data_shapes, meta_dict)
    output_shapes = summary_output_shapes_by_network(approximator, data_shapes, meta_dict)

    result = {0: input_shape}

    for k, v in output_shapes.items():
        if k != max(output_shapes.keys()):
            result[k + 1] = v

    return result


def data_condition_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data condition shapes of that inference network.
    """

    if not data_shapes:
        data_shapes = approximator.output_shapes

    inference_variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes, meta_dict)
    summary_output_shapes = summary_output_shapes_by_network(approximator, data_shapes, meta_dict)

    result = {}

    for network_idx, variable_shape in inference_variable_shapes.items():
        for k, v in summary_output_shapes.items():
            if variable_shape[:-1] == v[:-1]:
                result[network_idx] = replace_placeholders(v, meta_dict)

        if network_idx not in result:
            for k, v in summary_output_shapes.items():
                try:
                    concatenate_shapes([variable_shape, v])
                    result[network_idx] = replace_placeholders(variable_shape[:-1] + (v[-1],), meta_dict)
                except Exception:
                    pass

    return result


def inference_condition_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns the required inference condition shapes for each network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    output_shapes = resolve_shapes(data_shapes, meta_dict)
    data_node = approximator.graph.simulation_graph.data_node()

    data_condition_shapes = data_condition_shapes_by_network(approximator, data_shapes, meta_dict)
    summary_input = summary_input_shape(approximator, data_shapes, meta_dict)

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        condition_shapes = []
        for node in set(approximator.network_conditions[i]):
            if node in data_node:
                condition_shapes.append(data_condition_shapes[i])
            else:
                for variable in approximator.variable_names[node]:
                    shape = output_shapes[variable]

                    # flatten group dimension if node is not amortizable
                    if not approximator.graph.allows_amortization(node):
                        shape = shape[:-2] + (sp.prod(shape[-2:]),)

                    condition_shapes.append(shape)

        result[i] = list(concatenate_shapes(condition_shapes))
        result[i][-1] += len(summary_input[1:-1])  # add node repetitions
        result[i] = tuple(x for x in result[i])

    return result


def inference_variable_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that inference network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    output_shapes = resolve_shapes(data_shapes, meta_dict)

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        variable_shapes = []
        for node in approximator.network_composition[i]:
            for variable in approximator.variable_names[node]:
                variable_shapes.append(output_shapes[variable])

        result[i] = concatenate_shapes(variable_shapes)

    return result
