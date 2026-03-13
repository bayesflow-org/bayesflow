from typing import TYPE_CHECKING

import sympy as sp

from bayesflow.experimental.graphs import approximator_helpers

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator

from ..graphs.approximator_helpers import _summary_input_shape
from .shape_operations import replace_placeholders, resolve_shapes

Dim = int | sp.Expr
Shape = tuple[Dim, ...]


def summary_input_shape(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> tuple[int | sp.Expr, ...]:
    """
    Returns the input for the first summary network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes)

    input_shape = _summary_input_shape(approximator.graph, data_shapes, meta_dict)
    input_shape = replace_placeholders(input_shape, meta_dict)

    return input_shape


def summary_output_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, Shape]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that summary network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes)

    input_shapes = summary_input_shapes_by_network(approximator, data_shapes, meta_dict)
    output_shapes = {}

    for i, summary_net in enumerate(approximator.summary_networks or []):
        if all(isinstance(d, int) for d in input_shapes[i]):
            output_shapes[i] = summary_net.compute_output_shape(input_shapes[i])
        else:
            output_shapes[i] = input_shapes[i][:-2] + (sp.Symbol(f"summary_dim_{i}"),)

    return output_shapes


def summary_input_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are input shapes of that summary network.
    """

    if not data_shapes:
        data_shapes = approximator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    batch_size = next(iter(data_shapes.values()))[0]

    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes) | {"B": batch_size}

    input_shapes = approximator_helpers.summary_input_shapes_by_network(approximator.graph)
    input_shapes[0] = _summary_input_shape(approximator.graph, data_shapes)
    input_shapes = resolve_shapes(input_shapes, meta_dict)

    for i, summary_net in enumerate(approximator.summary_networks or []):
        if all(isinstance(d, int) for d in input_shapes[i]):
            output_shape = summary_net.compute_output_shape(input_shapes[i])
            meta_dict[sp.Symbol(f"summary_dim_{i}")] = output_shape[-1]
            input_shapes = resolve_shapes(input_shapes, meta_dict)

    return input_shapes


def data_condition_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data condition shapes of that inference network.
    """

    if not data_shapes:
        data_shapes = approximator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes)

    summary_output_shapes = summary_output_shapes_by_network(approximator, data_shapes, meta_dict)
    symbol_subs = {sp.Symbol(f"summary_dim_{k}"): v[-1] for k, v in summary_output_shapes.items()}

    result = approximator_helpers.data_condition_shapes_by_network(approximator.graph)
    result = resolve_shapes(result, meta_dict | symbol_subs)

    return result


def inference_condition_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: dict | None = None, meta_dict: dict | None = None
) -> dict[int, tuple[int | sp.Expr, ...]]:
    """
    Returns the required inference condition shapes for each network.
    """
    if not data_shapes:
        data_shapes = approximator.output_shapes

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes)

    summary_output_shapes = summary_output_shapes_by_network(approximator, data_shapes, meta_dict)
    symbol_subs = {sp.Symbol(f"summary_dim_{k}"): v[-1] for k, v in summary_output_shapes.items()}

    result = approximator_helpers.inference_condition_shapes_by_network(approximator.graph)
    result = resolve_shapes(result, meta_dict | symbol_subs)

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

    data_shapes = resolve_shapes(data_shapes, meta_dict)
    meta_dict = approximator._meta_dict_from_data_shapes(data_shapes)

    result = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)
    result = resolve_shapes(result, meta_dict)

    return result
