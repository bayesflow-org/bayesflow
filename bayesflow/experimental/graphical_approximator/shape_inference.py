from typing import TYPE_CHECKING

import sympy as sp

from bayesflow.experimental.graphs import approximator_helpers

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator

from .shape_operations import replace_placeholders, resolve_shapes
from ..graphs.approximator_helpers import _summary_input_shape

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

    input_shape = _summary_input_shape(approximator.graph)
    input_shape = replace_placeholders(input_shape, data_shapes)

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

    output_shapes = approximator_helpers.summary_output_shapes_by_network(approximator.graph)
    output_shapes = resolve_shapes(output_shapes, data_shapes)

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

    input_shapes = approximator_helpers.summary_input_shapes_by_network(approximator.graph)
    input_shapes = resolve_shapes(input_shapes, data_shapes)

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

    result = approximator_helpers.data_condition_shapes_by_network(approximator.graph)
    result = resolve_shapes(result, data_shapes)

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

    result = approximator_helpers.first_stage_condition_shapes_by_network(approximator.graph)
    result = resolve_shapes(result, data_shapes)

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

    result = approximator_helpers.inference_variable_shapes_by_network(approximator.graph)
    result = resolve_shapes(result, data_shapes)

    return result
