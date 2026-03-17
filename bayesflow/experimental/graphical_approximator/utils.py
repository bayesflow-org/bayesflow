from functools import reduce
from typing import TYPE_CHECKING, Mapping

from bayesflow.experimental.graphs.utils import sort_nodes_topologically

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator

import keras

from ...types import Shape, Tensor


def split_network_output(
    approximator: "GraphicalApproximator", output: Tensor, meta_dict: dict, network_idx: int
) -> dict[str, Tensor]:
    """
    Given the output of an inference network and its network index,
    splits the tensor into a dictionary where each key is a variable name
    and the values are tensors of the appropriate shape.
    """
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()
    output_shapes = approximator.graph.simulation_graph.output_shapes(meta_dict=meta_dict)

    samples = {}

    i = 0
    for node in network_composition[network_idx]:
        if approximator.graph.allows_amortization(node):
            # network already outputs a group dimension if there is one
            for variable in variable_names[node]:
                variable_dim = output_shapes[variable][-1]

                sample = output[..., i : (i + variable_dim)]
                samples[variable] = sample
                i += variable_dim
        else:
            # need to reshape so output has a group dimension
            for variable in variable_names[node]:
                variable_dim = output_shapes[variable][-1]
                group_dim = output_shapes[variable][-2]

                sample = output[..., i : (i + group_dim * variable_dim)]
                samples[variable] = keras.ops.expand_dims(sample, axis=-1)
                i += group_dim * variable_dim

    return samples


def summary_input(approximator: "GraphicalApproximator", data: Mapping) -> Tensor:
    """
    Returns the input for the first summary network.
    """
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    summary_input = concatenate([data[k] for k in data_keys])

    # permutate input so dimensions are put into summary networks in the required order
    shape_order = approximator.graph.data_shape_order()
    permutated_shape_order = approximator.graph.permutated_data_shape_order()
    indices = [shape_order.index(x) for x in permutated_shape_order]

    # indices does not refer to batch and data dimensions, so they have to be added
    indices = [0] + [idx + 1 for idx in indices] + list(range(len(indices) + 1, len(keras.ops.shape(summary_input))))

    return keras.ops.transpose(summary_input, axes=indices)


def summary_outputs_by_network(approximator: "GraphicalApproximator", data: Mapping) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are the outputs of that summary network.
    """
    input_tensor = summary_input(approximator, data)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        output_tensor = summary_network(input_tensor, training=False)
        result[i] = output_tensor

        input_tensor = output_tensor

    return result


def summary_inputs_by_network(approximator: "GraphicalApproximator", data: Mapping) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are the inputs of that summary network.
    """
    input_tensor = summary_input(approximator, data)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        result[i] = input_tensor
        output_tensor = summary_network(input_tensor, training=False)
        # next summary network uses previous output as input
        input_tensor = output_tensor

    return result


def data_conditions_by_network(approximator: "GraphicalApproximator", data: Mapping) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data conditions of that inference network.
    """
    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        result[i] = prepare_data_conditions(approximator, data, i)

    return result


def prepare_data_conditions(
    approximator: "GraphicalApproximator", data: Mapping, network_idx: int
) -> dict[int, Tensor]:
    """
    Returns the data conditions for the inference network denoted by `network_idx`.
    """
    summary_outputs = summary_outputs_by_network(approximator, data)
    required_dim = len(approximator.inference_networks[network_idx]._build_shapes_dict["xz_shape"])
    summary_by_dim = {len(keras.ops.shape(s)): s for s in summary_outputs.values()}

    return summary_by_dim[required_dim]


def inference_variables_by_network(approximator: "GraphicalApproximator", data: Mapping) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference variables of that inference network.
    """
    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        result[i] = prepare_inference_variables(approximator, data, i)

    return result


def prepare_inference_variables(approximator: "GraphicalApproximator", data: Mapping, network_idx: int) -> Tensor:
    """
    Returns the inference variables for the inference network denoted by `network_idx`.
    """
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()

    vars = []
    for node in network_composition[network_idx]:
        for name in variable_names[node]:
            var = data[name]

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

    return concatenate(vars)


def inference_conditions_by_network(approximator: "GraphicalApproximator", data: Mapping) -> dict[int, Tensor]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference conditions of that inference network.
    """
    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        result[i] = prepare_inference_conditions(approximator, data, i)

    return result


def prepare_inference_conditions(
    approximator: "GraphicalApproximator", data: Mapping, network_idx: int
) -> dict[int, Tensor]:
    """
    Returns the inference conditions for the inference network denoted by `network_idx`.
    """
    data_conditions = data_conditions_by_network(approximator, data)
    network_composition = approximator.graph.network_composition()[network_idx]
    network_conditions = approximator.graph.network_conditions()[network_idx]
    variable_names = approximator.graph.simulation_graph.variable_names()
    data_node = approximator.graph.simulation_graph.data_node()

    conditions = []
    nodes_to_condition_on = set(network_conditions) - {data_node} - set(network_composition)

    for node in sort_nodes_topologically(approximator.graph.simulation_graph, list(nodes_to_condition_on)):
        for name in variable_names[node]:
            var = data[name]

            # standardize conditions if required
            if name in approximator.standardize:
                var = approximator.standardize_layers[name](var, stage="validation")

            # flatten group dimension if node is not amortizable
            if not approximator.graph.allows_amortization(node):
                # transpose last two dimensions before flattening
                # so unpacking in split_network_output becomes easier
                rank = keras.ops.ndim(var)
                perm = (*range(rank - 2), rank - 1, rank - 2)
                transpose = keras.ops.transpose(var, axes=perm)

                var = keras.ops.reshape(transpose, (*keras.ops.shape(transpose)[:-2], -1))

            conditions.append(var)

    # add data conditions if necessary
    if data_conditions[network_idx] is not None:
        conditions.append(data_conditions[network_idx])

    conditions = concatenate(conditions)

    # add node repetitions
    repetitions = repetitions_from_data_shape(approximator, approximator._data_shapes(data))

    if repetitions != {}:
        conditions = add_node_reps_to_conditions(conditions, repetitions)

    return conditions


def add_node_reps_to_conditions(conditions, repetitions: Mapping[str, int]) -> Tensor:
    """
    Appends node repetition features (sqrt of node repetitons) to a conditions tensor.
    """
    rep_values = keras.ops.convert_to_tensor(list(repetitions.values()))
    squared = keras.ops.sqrt(rep_values)
    expanded = keras.ops.expand_dims(squared, axis=0)

    return concatenate([conditions, expanded])


def summary_input_shape(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]) -> Shape:
    """
    Returns the shape of the input tensor for the first summary network.
    """
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    input_shape = concatenate_shapes([data_shapes[k] for k in data_keys])

    # permutate input_shape so inputs are put into summary networks in the required order
    shape_order = approximator.graph.data_shape_order()
    permutated_shape_order = approximator.graph.permutated_data_shape_order()
    indices = [shape_order.index(x) for x in permutated_shape_order]

    # indices does not refer to batch and data dimensions, so they have to be added
    indices = [0] + [idx + 1 for idx in indices] + list(range(len(indices) + 1, len(input_shape)))
    input_shape = keras.ops.convert_to_tensor([input_shape[idx] for idx in indices])

    return input_shape


def summary_output_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[int, Shape]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that summary network.
    """
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        output_shape = summary_network.compute_output_shape(tuple(shape))
        result[i] = keras.ops.convert_to_tensor(output_shape)

        # next summary network uses previous output as input
        input_shape = tuple(output_shape)

    return result


def summary_input_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[int, Shape]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are input shapes of that summary network.
    """
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        result[i] = tuple(int(i) for i in keras.ops.convert_to_tensor(tuple(shape)))

        output_shape = summary_network.compute_output_shape(tuple(shape))

        # next summary network uses previous output as input
        input_shape = tuple(output_shape)

    return result


def data_condition_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[int, Shape]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data condition shapes of that inference network.
    """
    inference_shapes = inference_variable_shapes_by_network(approximator, data_shapes)
    conditions = approximator.graph.network_conditions()
    data_node = approximator.graph.simulation_graph.data_node()

    summary_shapes = summary_output_shapes_by_network(approximator, data_shapes)
    summary_by_dim = {len(s): s for s in summary_shapes.values()}

    result = {}

    for i, variable_shape in inference_shapes.items():
        # data dimension must be identical to inference variable dimension
        dim = len(variable_shape)

        # only add data conditions if data node is in network conditions
        if data_node in conditions[i]:
            result[i] = summary_by_dim[dim]

    return result


def inference_variable_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[int, Shape]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that inference network.
    """
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        variable_shapes = []
        for node in network_composition[i]:
            for variable in variable_names[node]:
                shape = data_shapes[variable]

                # flatten group dimension if node is not amortizable
                if not approximator.graph.allows_amortization(node):
                    shape = shape[:-2] + (int(keras.ops.prod(shape[-2:])),)

                variable_shapes.append(tuple(shape))

        result[i] = tuple(int(i) for i in concatenate_shapes(variable_shapes))

    return result


def inference_variable_shapes_from_meta(approximator: "GraphicalApproximator", meta: dict) -> dict[int, Shape]:
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that inference network.
    The first index (batch_size) of the output shapes is always 1.
    """
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()
    output_shapes = approximator.graph.simulation_graph.output_shapes(meta)

    result = {}

    # add batch size to output_shapes
    for k, v in output_shapes.items():
        output_shapes[k] = (1,) + v[1:]

    for i, _ in enumerate(approximator.inference_networks):
        variable_shapes = []
        for node in network_composition[i]:
            for variable in variable_names[node]:
                shape = output_shapes[variable]

                # flatten group dimension if node is not amortizable
                if not approximator.graph.allows_amortization(node):
                    shape = shape[:-2] + (int(keras.ops.prod(shape[-2:])),)

                variable_shapes.append(tuple(shape))

        result[i] = tuple(int(i) for i in concatenate_shapes(variable_shapes))

    return result


def inference_condition_shapes_by_network(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[int, Shape]:
    """
    Returns the required inference condition shapes for each network.
    """
    data_conditions = data_condition_shapes_by_network(approximator, data_shapes)
    network_composition = approximator.graph.network_composition()
    network_conditions = approximator.graph.network_conditions()
    variable_names = approximator.graph.simulation_graph.variable_names()
    data_node = approximator.graph.simulation_graph.data_node()
    repetitions = repetitions_from_data_shape(approximator, data_shapes)

    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        # collect shapes from all variables in the nodes
        condition_shapes = []
        nodes_to_condition_on = set(network_conditions[i]) - {data_node} - set(network_composition[i])

        for node in nodes_to_condition_on:
            for variable in variable_names[node]:
                shape = data_shapes[variable]

                # flatten group dimension if node is not amortizable
                if not approximator.graph.allows_amortization(node):
                    shape = shape[:-2] + (int(keras.ops.prod(shape[-2:])),)

                condition_shapes.append(tuple(shape))

        # add data conditions if necessary
        if data_conditions[i] is not None:
            condition_shapes.append(data_conditions[i])

        concatenated = list(concatenate_shapes(condition_shapes))

        # add all node repetition to all conditions.
        # For some nodes, the number of conditions could be further reduced, but this would
        # require additional logic.
        concatenated[-1] += len(repetitions)
        result[i] = tuple(int(i) for i in concatenated)

    return result


def meta_dict_from_data_shapes(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[str, ...]:
    """
    Infers meta information from data shapes.
    """
    meta_dict = {}
    output_shapes = approximator.graph.simulation_graph.output_shapes()

    for k, v in data_shapes.items():
        for dim_a, dim_b in zip(output_shapes[k], v):
            if isinstance(dim_a, str):
                meta_dict[dim_a] = dim_b

    return meta_dict


def repetitions_from_data_shape(
    approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]
) -> dict[str, int]:
    """
    Infers repetition counts for each node from data shapes.
    """
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    summary_input_shape = concatenate_shapes([data_shapes[k] for k in data_keys])
    shape_order = approximator.graph.data_shape_order()

    repetitions = {}

    for i, variable in enumerate(shape_order):
        repetitions[variable] = summary_input_shape[1:-1][i]  # skip batch and and variable dimension

    return repetitions


def concatenate(tensors, batch_dims=1):
    if keras.backend.backend() == "tensorflow":
        return concatenate_tf(tensors, batch_dims=batch_dims)
    else:
        return concatenate_(tensors, batch_dims=batch_dims)


def concatenate_tf(tensors, batch_dims=1):
    def _shape_tensor(t):
        """
        Special case for tensorflow, because keras.ops.shape(t) returns a Python tuple.
        """
        s = keras.ops.shape(t)
        return keras.ops.convert_to_tensor(s, dtype="int32")

    max_rank = max(len(t.shape) for t in tensors)

    expanded = []
    for t in tensors:
        while len(t.shape) < max_rank:
            t = keras.ops.expand_dims(t, axis=-2)
        expanded.append(t)

    shapes = [_shape_tensor(t) for t in expanded]
    base = shapes[0]

    if max_rank - 1 > batch_dims:
        mids = keras.ops.stack(
            [s[batch_dims : max_rank - 1] for s in shapes],
            axis=0,
        )

        mids_max = keras.ops.max(mids, axis=0)
        max_shape = keras.ops.concatenate(
            [base[:batch_dims], mids_max, base[max_rank - 1 : max_rank]],
            axis=0,
        )
    else:
        max_shape = base

    broadcasted = []
    for t, s in zip(expanded, shapes):
        target = keras.ops.concatenate(
            [
                max_shape[: max_rank - 1],
                s[max_rank - 1 : max_rank],
            ],
            axis=0,
        )

        broadcasted.append(keras.ops.broadcast_to(t, target))

    return keras.ops.concatenate(broadcasted, axis=-1)


def concatenate_(tensors, batch_dims=1) -> Tensor:
    max_rank = max(len(t.shape) for t in tensors)

    expanded = []
    for t in tensors:
        while len(t.shape) < max_rank:
            t = keras.ops.expand_dims(t, axis=-2)
        expanded.append(t)

    # compute max shape
    max_shape = list(expanded[0].shape)
    for t in expanded[1:]:
        for i in range(batch_dims, max_rank - 1):
            max_shape[i] = max(max_shape[i], t.shape[i])

    max_shape[0] = keras.ops.shape(tensors[0])[0]
    broadcasted = []
    for t in expanded:
        # keep last dimension unique
        target = tuple(max_shape[:-1] + [t.shape[-1]])
        broadcasted.append(keras.ops.broadcast_to(t, target))

    return keras.ops.concatenate(broadcasted, axis=-1)


def add_sample_dimension(tensor: Tensor, num_samples: int, batch_dims: int = 1) -> Tensor:
    """
    Introduces a sample dimension right after batch_dims dimensions.

    >>> x = keras.random.normal((10, 5))
    >>> y = add_sample_dimension(x, 55)
    >>> keras.ops.shape(y)
    (10, 55, 5)
    """
    shape = keras.ops.shape(tensor)
    target_shape = (*shape[:batch_dims], num_samples, *shape[batch_dims:])

    expanded = keras.ops.expand_dims(tensor, axis=batch_dims)
    stacked = keras.ops.broadcast_to(expanded, target_shape)

    return stacked


def concatenate_shapes(shapes) -> Tensor:
    max_rank = max(len(s) for s in shapes)
    expanded = [expand_shape_rank(s, max_rank) for s in shapes]

    return reduce(stack_shapes, expanded)


def stack_shapes(a, b, axis=-1) -> Tensor:
    rank = max(keras.ops.shape(a)[0], keras.ops.shape(b)[0])
    a = expand_shape_rank(a, rank)
    a = keras.ops.convert_to_tensor(a)

    b = expand_shape_rank(b, rank)
    b = keras.ops.convert_to_tensor(b)

    sum_tensor = a + b
    max_tensor = keras.ops.maximum(a, b)
    axis = axis % rank

    idx = keras.ops.arange(rank)
    mask = idx == axis

    shape = keras.ops.where(mask, sum_tensor, max_tensor)

    return shape


def expand_shape_rank(shape, target_rank) -> Tensor:
    """
    Expand a tensor shape to a desired rank by inserting singleton (1)
    dimensions immediately before the last dimension.
    >>> expand_shape_rank((10, 2, 3), 5)
    (10, 2, 1, 1, 3)
    """
    shape = keras.ops.convert_to_tensor(shape)
    n = target_rank - keras.ops.shape(shape)[0]
    shape_rank = keras.ops.concatenate(
        [keras.ops.convert_to_tensor(shape[:-1]), keras.ops.ones((n,)), keras.ops.take(shape, indices=[-1], axis=0)],
        axis=0,
    )
    shape_rank = keras.ops.cast(shape_rank, "int")

    return shape_rank
