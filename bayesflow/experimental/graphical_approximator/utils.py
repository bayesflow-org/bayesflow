from functools import reduce
from typing import TYPE_CHECKING, Mapping

from bayesflow.experimental.graphs.utils import sort_nodes_topologically

if TYPE_CHECKING:
    from .graphical_approximator import GraphicalApproximator

import keras

from ...types import Shape, Tensor


def split_network_output(approximator: "GraphicalApproximator", output: Tensor, network_idx: int):
    """
    Given the output of an inference network and its network index,
    splits the tensor into a dictionary where each key is a variable name
    and the values are tensors of the appropriate shape.
    """
    network_composition = approximator.graph.network_composition()
    variable_names = approximator.graph.simulation_graph.variable_names()

    samples = {}

    i = 0
    for node in network_composition[network_idx]:
        if approximator.graph.allows_amortization(node):
            # network already outputs a group dimension if there is one
            for variable in variable_names[node]:
                variable_dim = approximator.data_shapes[variable][-1]

                sample = output[..., i : (i + variable_dim)]
                samples[variable] = sample
                i += variable_dim
        else:
            # need to reshape so output has a group dimension
            for variable in variable_names[node]:
                variable_dim = approximator.data_shapes[variable][-1]
                group_dim = approximator.data_shapes[variable][-2]

                sample = output[..., i : (i + group_dim * variable_dim)]
                samples[variable] = keras.ops.expand_dims(sample, axis=-1)
                i += group_dim * variable_dim

    return samples


def summary_input(approximator: "GraphicalApproximator", data: Mapping):
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


def summary_outputs_by_network(approximator: "GraphicalApproximator", data: Mapping):
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


def summary_inputs_by_network(approximator: "GraphicalApproximator", data: Mapping):
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


def data_conditions_by_network(approximator: "GraphicalApproximator", data: Mapping):
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are data conditions of that inference network.
    """
    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        result[i] = prepare_data_conditions(approximator, data, i)

    return result


def prepare_data_conditions(approximator: "GraphicalApproximator", data: Mapping, network_idx: int):
    """
    Returns the data conditions for the inference network denoted by `network_idx`.
    """
    conditions = approximator.graph.network_conditions()[network_idx]
    data_node = approximator.graph.simulation_graph.data_node()

    if data_node not in conditions:
        return None

    summary_outputs = summary_outputs_by_network(approximator, data)
    required_dim = len(approximator.inference_networks[network_idx].base_distribution.dims) + 1
    summary_by_dim = {len(keras.ops.shape(s)): s for s in summary_outputs.values()}

    return summary_by_dim[required_dim]


def inference_variables_by_network(approximator: "GraphicalApproximator", data: Mapping):
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference variables of that inference network.
    """
    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        result[i] = prepare_inference_variables(approximator, data, i)

    return result


def prepare_inference_variables(approximator: "GraphicalApproximator", data: Mapping, network_idx: int):
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


def inference_conditions_by_network(approximator: "GraphicalApproximator", data: Mapping):
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are inference conditions of that inference network.
    """
    result = {}

    for i, _ in enumerate(approximator.inference_networks):
        result[i] = prepare_inference_conditions(approximator, data, i)

    return result


def prepare_inference_conditions(approximator: "GraphicalApproximator", data: Mapping, network_idx: int):
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
                var = approximator.standardize_layers[name](var, staging="validation")

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


def add_node_reps_to_conditions(conditions, repetitions: Mapping[str, int]):
    """
    Appends node repetition features to a conditions tensor.
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
    input_shape = tuple(input_shape[idx] for idx in indices)

    return input_shape


def summary_output_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]):
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are output shapes of that summary network.
    """
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        output_shape = summary_network.compute_output_shape(shape)
        result[i] = output_shape

        # next summary network uses previous output as input
        input_shape = output_shape

    return result


def summary_input_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]):
    """
    Returns a dictionary where the keys are integers denoting network indices
    and the values are input shapes of that summary network.
    """
    input_shape = summary_input_shape(approximator, data_shapes)

    result = {}

    for i, summary_network in enumerate(approximator.summary_networks or []):
        shape = input_shape + (1,) if len(input_shape) == 2 else input_shape
        result[i] = input_shape

        output_shape = summary_network.compute_output_shape(shape)

        # next summary network uses previous output as input
        input_shape = output_shape

    return result


def data_condition_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]):
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
        else:
            result[i] = None

    return result


def inference_variable_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]):
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

        result[i] = tuple(concatenate_shapes(variable_shapes).numpy().tolist())

    return result


def inference_condition_shapes_by_network(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]):
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
        result[i] = tuple(keras.ops.convert_to_numpy(concatenated).tolist())

    return result


def repetitions_from_data_shape(approximator: "GraphicalApproximator", data_shapes: Mapping[str, Shape]):
    """
    Infers repetition counts for each node from data shapes.
    """
    data_node = approximator.graph.simulation_graph.data_node()
    data_keys = approximator.graph.simulation_graph.variable_names()[data_node]

    summary_input_shape = concatenate_shapes([data_shapes[k] for k in data_keys])
    shape_order = approximator.graph.data_shape_order()

    repetitions = {}

    # looping in reverse for easier indexing
    for i, variable in enumerate(shape_order):
        repetitions[variable] = summary_input_shape[i - len(shape_order) - 1]

    return repetitions


def concatenate(tensors, batch_dims=1):
    """
    Concatenates tensors of possibly unequal ranks by expanding and
    tiling missing dimensions.

    >>> x = keras.random.normal((20, 5))
    >>> y = keras.random.normal((20, 15, 3)
    >>> z = concatenate([x, y])
    >>> keras.ops.shape(z)
    (20, 15, 8)
    """
    max_rank = max([len(keras.ops.shape(t)) for t in tensors])

    # expand tensors so each tensor has rank max_rank
    expanded = []

    for t in tensors:
        flat_shape = (-1, *keras.ops.shape(t)[batch_dims:])
        flat = keras.ops.reshape(t, flat_shape)

        expanded_shape = expand_shape_rank(keras.ops.shape(flat), max_rank)
        expanded.append(keras.ops.reshape(flat, expanded_shape))

    # compute max size along each dimension
    expanded_shapes = [keras.ops.shape(t) for t in expanded]
    max_shape_per_dim = keras.ops.max(keras.ops.stack(expanded_shapes, axis=0), axis=0)

    # broadcast tensors to match max_shape
    target_shapes = [
        keras.ops.concatenate([max_shape_per_dim[:-1], keras.ops.convert_to_tensor(keras.ops.shape(t)[-1:])], axis=0)
        for t in expanded
    ]  #

    # target_shapes = [(*max_shape_per_dim[:-1], keras.ops.shape(t)[-1]) for t in expanded]
    broadcasted = [keras.ops.broadcast_to(t, s) for t, s in zip(expanded, target_shapes)]

    # concatenate along last dimension
    concatenated = keras.ops.concatenate(broadcasted, axis=-1)

    # restore original batch dimensions
    original_batch_shape = keras.ops.shape(tensors[0])[:batch_dims]
    final_shape = (*original_batch_shape, *keras.ops.shape(concatenated)[1:])

    return keras.ops.reshape(concatenated, final_shape)


def add_sample_dimension(tensor, num_samples, batch_dims=1):
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


def concatenate_shapes(shapes):
    """
    Concatenate shapes by expanding them to the same rank and then
    summing sizes along the last axis.

    >>> concatenate_shapes([(7, 5, 2), (3, 20)])
    <keras.Tensor shape=(3,) dtype=int32 values=[7, 5, 22]>
    """
    ranks = keras.ops.stack([keras.ops.size(s) for s in shapes], axis=0)
    max_rank = keras.ops.max(ranks)

    expanded = [expand_shape_rank(s, max_rank) for s in shapes]

    return reduce(stack_shapes, expanded)


def stack_shapes(a, b, axis=-1):
    """
    Compute the resulting shape of stacking two tensors along a given axis.

    >>> stack_shapes((10, 2, 3), (32, 1))
    <keras.Tensor shape=(3,) dtype=int32 values=[32, 2, 4]>
    """
    a = keras.ops.convert_to_tensor(a, dtype="int32")
    b = keras.ops.convert_to_tensor(b, dtype="int32")

    rank = keras.ops.maximum(keras.ops.size(a), keras.ops.size(b))
    a = keras.ops.convert_to_tensor(expand_shape_rank(a, rank), dtype="int32")
    b = keras.ops.convert_to_tensor(expand_shape_rank(b, rank), dtype="int32")

    axis = keras.ops.convert_to_tensor(axis, dtype="int32")
    axis = keras.ops.where(axis < 0, axis + rank, axis)

    i = keras.ops.arange(rank, dtype="int32")
    stacked = keras.ops.where(i == axis, a + b, keras.ops.maximum(a, b))

    return stacked


def expand_shape_rank(shape, target_rank):
    """
    Expand a tensor shape to a desired rank by inserting singleton (1)
    dimensions immediately before the last dimension.

    >>> expand_shape_rank((10, 2, 3), 5)
    <keras.Tensor shape=(5,) dtype=int32 values=[10, 2, 1, 1, 3]>
    """

    shape = keras.ops.convert_to_tensor(shape)
    target_rank = keras.ops.cast(target_rank, dtype="int32")

    shape = keras.ops.cast(shape, dtype="int32")
    rank = keras.ops.size(shape)

    k = keras.ops.maximum(target_rank - rank, 0)

    prefix = shape[:-1]
    last = shape[-1:]
    ones = keras.ops.ones([k], dtype=shape.dtype)

    expanded = keras.ops.concatenate([prefix, ones, last], axis=0)

    return expanded
