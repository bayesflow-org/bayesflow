from functools import reduce
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator


def summary_input_shape(approximator: "GraphicalApproximator", meta_dict: dict | None = None) -> tuple[int | str, ...]:
    """
    Returns the input for the first summary network.
    """
    output_shapes = simulator_output_shapes(approximator, meta_dict)
    data_node = approximator.graph.simulation_graph.data_node()
    data_variables = approximator.graph.simulation_graph.variable_names()[data_node]

    shapes = []

    for variable in data_variables:
        shapes.append(output_shapes[variable])

    return concatenate_shapes(shapes)


def summary_output_shapes_by_network(
    approximator: "GraphicalApproximator", meta_dict: dict | None = None
) -> dict[int, tuple[int | str, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that summary network.
    """
    input_shape = summary_input_shape(approximator, meta_dict)

    placeholders = [x for x in input_shape if isinstance(x, str)]
    mock_meta = {k: 1 for k in placeholders} | {"B": 1}
    mock_input_shape = replace_placeholders(input_shape, mock_meta)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        output_shape = summary_network.compute_output_shape(mock_input_shape)
        result[i] = input_shape[: len(output_shape) - 1] + (output_shape[-1],)
        mock_input_shape = mock_input_shape[: len(output_shape) - 1] + (output_shape[-1],)

    return result


def data_condition_shapes_by_network(
    approximator: "GraphicalApproximator", meta_dict: dict | None = None
) -> dict[int, tuple[int | str, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data condition shapes of that inference network.
    """

    inference_variable_shapes = inference_variable_shapes_by_network(approximator)
    summary_output_shapes = summary_output_shapes_by_network(approximator)

    result = {}

    for network_idx, variable_shape in inference_variable_shapes.items():
        for k, v in summary_output_shapes.items():
            if variable_shape[:-1] == v[:-1]:
                result[network_idx] = replace_placeholders(v, meta_dict)

        if network_idx not in result:
            for k, v in summary_output_shapes.items():
                try:
                    concatenate_shapes([variable_shape, v])
                    result[network_idx] = replace_placeholders(v, meta_dict)
                except Exception:
                    pass

    return result


def inference_condition_shapes_by_network(
    approximator: "GraphicalApproximator", meta_dict: dict | None = None
) -> dict[int, tuple[int | str, ...]]:
    """
    Returns the required inference condition shapes for each network.
    """
    output_shapes = simulator_output_shapes(approximator, meta_dict)
    network_conditions = approximator.graph.network_conditions()
    variable_names = approximator.graph.simulation_graph.variable_names()
    data_node = approximator.graph.simulation_graph.data_node()
    data_condition_shapes = data_condition_shapes_by_network(approximator, meta_dict)

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        condition_shapes = []
        for node in set(network_conditions[i]) - set(data_node):
            for variable in variable_names[node]:
                condition_shapes.append(output_shapes[variable])

            if node in data_node:
                condition_shapes.append(data_condition_shapes[i])

    return result


def inference_variable_shapes_by_network(
    approximator: "GraphicalApproximator", meta_dict: dict | None = None
) -> dict[int, tuple[int | str, ...]]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that inference network.
    """
    output_shapes = simulator_output_shapes(approximator, meta_dict)
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        variable_shapes = []
        for node in network_composition[i]:
            for variable in variable_names[node]:
                variable_shapes.append(output_shapes[variable])

        result[i] = concatenate_shapes(variable_shapes)

    return result


def simulator_output_shapes(approximator: "GraphicalApproximator", meta_dict: dict | None = None):
    """
    Returns the output shape of each simulated variable in the simulation graph.
    When no `meta_dict` is supplied, the output shapes contain placeholders for
    node repetitions.
    """
    return approximator.graph.simulation_graph.output_shapes(meta_dict=meta_dict)


def concatenate_shapes(shapes: list[tuple[int | str, ...]]) -> tuple[int | str, ...]:
    """
    Combine multiple shapes into a single stacked shape.
    All input shapes are first expanded to the same rank, then merged
    by iteratively stacking them along the last axis.
    """
    max_rank = max(len(s) for s in shapes)
    expanded = [expand_shape_rank(s, max_rank) for s in shapes]

    return reduce(stack_shapes, expanded)


def replace_placeholders(shape: tuple[int | str, ...], meta_dict: dict | None = None) -> tuple[int | str, ...]:
    """
    Replaces string placeholders in shape tuples.
    >>> replace_placeholders(("B", "N", 3), {"B": 4, "N": 3})
    (4, 3, 3)
    """

    meta_dict = meta_dict or {}

    def resolve(x: str | int) -> int | str:
        if x in meta_dict:
            return meta_dict[x]
        else:
            return x

    return tuple(resolve(x) for x in shape)


def stack_shapes(a: tuple[int | str, ...], b: tuple[int | str, ...], axis=-1) -> tuple[int | str, ...]:
    """
    Stacks two shape tuples along an axis. The size along the staking axis
    is the sum of both shapes, while all other dimensions take the maximum of
    the two.
    >>> stack_shapes((2, 20), (2, 3, 3), axis=-1)
    (2, 3, 23)
    """
    rank = max(len(a), len(b))
    axis = axis % rank

    a = expand_shape_rank(a, rank)
    b = expand_shape_rank(b, rank)

    output = []
    for i in range(rank):
        if isinstance(a[i], str):
            if b[i] == a[i] or b[i] == 1:
                output.append(a[i])
                continue
        if isinstance(b[i], str):
            if a[i] == b[i] or a[i] == 1:
                output.append(b[i])
                continue

        if isinstance(a[i], str) or isinstance(b[i], str):
            raise ValueError(f"Cannot stack shapes with differing placeholders {a[i]} and {b[i]}.")
        else:
            output.append(a[i] + b[i] if i == axis else max(a[i], b[i]))  # type: ignore

    return tuple(output)


def expand_shape_rank(shape: tuple[int | str, ...], target_rank: int) -> tuple[int | str, ...]:
    """
    Expand a tensor shape to a desired rank by inserting singleton (1)
    dimensions immediately before the last dimension.
    >>> expand_shape_rank((10, 2, 3), 5)
    (10, 2, 1, 1, 3)
    """

    rank = len(shape)
    if target_rank < rank:
        raise ValueError(f"Shape rank {len(shape)} must be less than target_rank {target_rank}.")

    inserts = target_rank - rank

    return shape[:-1] + (1,) * inserts + shape[-1:]
