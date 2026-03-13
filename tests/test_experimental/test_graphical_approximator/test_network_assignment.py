import keras

from bayesflow.experimental.graphical_approximator.tensor_concatenation import concatenate


def test_inference_variables_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import inference_variables_by_network
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    beta = approximator.standardize_layers["beta"](data["beta"])
    sigma = approximator.standardize_layers["sigma"](data["sigma"])

    expected_output = concatenate([beta, sigma])
    assert keras.ops.all(variables[0] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variables_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import inference_variables_by_network
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    data = two_level_simulator.sample(2)
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    hyper_mean = approximator.standardize_layers["hyper_mean"](data["hyper_mean"])
    hyper_std = approximator.standardize_layers["hyper_std"](data["hyper_std"])
    shared_std = approximator.standardize_layers["shared_std"](data["shared_std"])

    expected_output = concatenate([hyper_mean, hyper_std, shared_std])
    assert keras.ops.all(variables[0] == expected_output)

    expected_output = approximator.standardize_layers["local_mean"](data["local_mean"])
    assert keras.ops.all(variables[1] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variables_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import inference_variables_by_network
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    school_mu = approximator.standardize_layers["school_mu"](data["school_mu"])
    school_sigma = approximator.standardize_layers["school_sigma"](data["school_sigma"])
    shared_sigma = approximator.standardize_layers["shared_sigma"](data["shared_sigma"])

    expected_output = concatenate([school_mu, school_sigma, shared_sigma])
    assert keras.ops.all(variables[0] == expected_output)

    classroom_mu = approximator.standardize_layers["classroom_mu"](data["classroom_mu"])
    classroom_sigma = approximator.standardize_layers["classroom_sigma"](data["classroom_sigma"])

    expected_output = concatenate([classroom_mu, classroom_sigma])
    assert keras.ops.all(variables[1] == expected_output)

    student_mu = approximator.standardize_layers["student_mu"](data["student_mu"])
    student_sigma = approximator.standardize_layers["student_sigma"](data["student_sigma"])

    expected_output = concatenate([student_mu, student_sigma])
    assert keras.ops.all(variables[2] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variables_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.network_assignment import inference_variables_by_network
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    mu_question_mean = approximator.standardize_layers["mu_question_mean"](data["mu_question_mean"])
    sigma_question_mean = approximator.standardize_layers["sigma_question_mean"](data["sigma_question_mean"])
    mu_question_std = approximator.standardize_layers["mu_question_std"](data["mu_question_std"])
    sigma_question_std = approximator.standardize_layers["sigma_question_std"](data["sigma_question_std"])

    expected_output = concatenate([mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std])
    assert keras.ops.all(variables[0] == expected_output)

    question_mean = approximator.standardize_layers["question_mean"](data["question_mean"])
    question_std = approximator.standardize_layers["question_std"](data["question_std"])
    question_difficulty = approximator.standardize_layers["question_difficulty"](data["question_difficulty"])

    expected_output = concatenate([question_mean, question_std, question_difficulty])
    assert keras.ops.all(variables[1] == expected_output)

    expected_output = approximator.standardize_layers["student_ability"](data["student_ability"])
    assert keras.ops.all(variables[2] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_summary_input_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import summary_input
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = single_level_approximator.adapter(single_level_simulator.sample(2))
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    expected_output = concatenate([data["x"], data["y"]])
    observed = summary_input(approximator, data)

    assert keras.ops.all(observed == expected_output)

    input_shape = summary_input_shape(approximator, data_shapes)
    assert observed.shape == input_shape


def test_summary_input_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import summary_input
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = two_level_approximator.adapter(two_level_simulator.sample(2))
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    expected_output = data["y"]
    observed = summary_input(approximator, data)
    assert keras.ops.all(keras.ops.isclose(observed, expected_output))

    input_shape = summary_input_shape(approximator, data_shapes)
    assert observed.shape == input_shape


def test_summary_input_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import summary_input
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = three_level_approximator.adapter(three_level_simulator.sample(2))
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    expected_output = data["y"]
    observed = summary_input(approximator, data)
    assert keras.ops.all(keras.ops.isclose(observed, expected_output))

    input_shape = summary_input_shape(approximator, data_shapes)
    assert observed.shape == input_shape


def test_summary_input_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import summary_input
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(2))
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    expected_output = data["obs"]
    observed = summary_input(approximator, data)
    assert keras.ops.all(keras.ops.isclose(observed, expected_output))

    input_shape = summary_input_shape(approximator, data_shapes)
    assert observed.shape == input_shape


def test_summary_outputs_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = single_level_approximator.adapter(single_level_simulator.sample(2))
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    input = summary_input(approximator, data)

    assert keras.ops.all(summary_outputs[0] == approximator.summary_networks[0](input))

    output_shapes = summary_output_shapes_by_network(approximator, data_shapes)

    for k, v in summary_outputs.items():
        assert v.shape == output_shapes[k]


def test_summary_outputs_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = two_level_approximator.adapter(two_level_simulator.sample(2))
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    input = summary_input(approximator, data)

    assert keras.ops.all(summary_outputs[0] == approximator.summary_networks[0](input))
    assert keras.ops.all(summary_outputs[1] == approximator.summary_networks[1](summary_outputs[0]))

    output_shapes = summary_output_shapes_by_network(approximator, data_shapes)

    for k, v in summary_outputs.items():
        assert v.shape == output_shapes[k]


def test_summary_outputs_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = three_level_approximator.adapter(three_level_simulator.sample(2))
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    input = summary_input(approximator, data)

    assert keras.ops.all(summary_outputs[0] == approximator.summary_networks[0](input))
    assert keras.ops.all(summary_outputs[1] == approximator.summary_networks[1](summary_outputs[0]))
    assert keras.ops.all(summary_outputs[2] == approximator.summary_networks[2](summary_outputs[1]))

    output_shapes = summary_output_shapes_by_network(approximator, data_shapes)

    for k, v in summary_outputs.items():
        assert v.shape == output_shapes[k]


def test_summary_outputs_by_network_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(2))
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    input = summary_input(approximator, data)

    assert keras.ops.all(summary_outputs[0] == approximator.summary_networks[0](input))
    assert keras.ops.all(summary_outputs[1] == approximator.summary_networks[1](summary_outputs[0]))

    output_shapes = summary_output_shapes_by_network(approximator, data_shapes)

    for k, v in summary_outputs.items():
        assert v.shape == output_shapes[k]


def test_summary_inputs_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_inputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = single_level_approximator.adapter(single_level_simulator.sample(2))
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    summary_inputs = summary_inputs_by_network(approximator, data)

    assert keras.ops.all(summary_inputs[0] == summary_input(approximator, data))

    input_shapes = summary_input_shapes_by_network(approximator, data_shapes)

    for k, v in summary_inputs.items():
        assert v.shape == input_shapes[k]


def test_summary_inputs_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_inputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = two_level_approximator.adapter(two_level_simulator.sample(2))
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    summary_inputs = summary_inputs_by_network(approximator, data)

    assert keras.ops.all(summary_inputs[0] == summary_input(approximator, data))
    assert keras.ops.all(summary_inputs[1] == approximator.summary_networks[0](summary_inputs[0]))

    input_shapes = summary_input_shapes_by_network(approximator, data_shapes)

    for k, v in summary_inputs.items():
        assert v.shape == input_shapes[k]


def test_summary_inputs_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_inputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = three_level_approximator.adapter(three_level_simulator.sample(2))
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    summary_inputs = summary_inputs_by_network(approximator, data)

    assert keras.ops.all(summary_inputs[0] == summary_input(approximator, data))
    assert keras.ops.all(summary_inputs[1] == approximator.summary_networks[0](summary_inputs[0]))
    assert keras.ops.all(summary_inputs[2] == approximator.summary_networks[1](summary_inputs[1]))

    input_shapes = summary_input_shapes_by_network(approximator, data_shapes)

    for k, v in summary_inputs.items():
        assert v.shape == input_shapes[k]


def test_summary_inputs_by_network_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        summary_input,
        summary_inputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = crossed_design_irt_approximator.adapter(crossed_design_irt_simulator.sample(2))
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    summary_inputs = summary_inputs_by_network(approximator, data)

    assert keras.ops.all(summary_inputs[0] == summary_input(approximator, data))
    assert keras.ops.all(summary_inputs[1] == approximator.summary_networks[0](summary_inputs[0]))

    input_shapes = summary_input_shapes_by_network(approximator, data_shapes)

    for k, v in summary_inputs.items():
        assert v.shape == input_shapes[k]


def test_data_conditions_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        data_conditions_by_network,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    conditions = data_conditions_by_network(approximator, data)

    assert keras.ops.all(conditions[0] == summary_outputs[0])

    condition_shapes = data_condition_shapes_by_network(approximator, data_shapes)

    for k, v in conditions.items():
        assert v.shape == condition_shapes[k]


def test_data_conditions_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        data_conditions_by_network,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = two_level_simulator.sample(2)
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    conditions = data_conditions_by_network(approximator, data)

    assert keras.ops.all(conditions[0] == summary_outputs[1])
    assert keras.ops.all(conditions[1] == summary_outputs[0])

    condition_shapes = data_condition_shapes_by_network(approximator, data_shapes)

    for k, v in conditions.items():
        assert v.shape == condition_shapes[k]


def test_data_conditions_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        data_conditions_by_network,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    conditions = data_conditions_by_network(approximator, data)

    assert keras.ops.all(conditions[0] == summary_outputs[2])
    assert keras.ops.all(conditions[1] == summary_outputs[1])
    assert keras.ops.all(conditions[2] == summary_outputs[0])

    condition_shapes = data_condition_shapes_by_network(approximator, data_shapes)

    for k, v in conditions.items():
        assert v.shape == condition_shapes[k]


def test_data_conditions_by_network_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        data_conditions_by_network,
        summary_outputs_by_network,
    )
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    summary_outputs = summary_outputs_by_network(approximator, data)
    conditions = data_conditions_by_network(approximator, data)

    assert keras.ops.all(conditions[0] == summary_outputs[1])
    assert keras.ops.all(conditions[1] == summary_outputs[0])
    assert keras.ops.all(conditions[2] == summary_outputs[2])

    condition_shapes = data_condition_shapes_by_network(approximator, data_shapes)

    for k, v in conditions.items():
        assert v.shape == condition_shapes[k]


def test_inference_conditions_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        inference_conditions_by_network,
        summary_input,
        summary_outputs_by_network,
    )

    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data)

    # output has the correct shape
    expected_shape = (2, 11)
    assert keras.ops.shape(conditions[0]) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[0]

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate([data_conditions, expanded])

    assert keras.ops.all(conditions[0] == expected_output)


def test_inference_conditions_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        inference_conditions_by_network,
        summary_input,
        summary_outputs_by_network,
    )

    data = two_level_simulator.sample(2)
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data)

    # output has the correct shape
    expected_shape = (2, 22)
    assert keras.ops.shape(conditions[0]) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[0]

    hyper_mean = approximator.standardize_layers["hyper_mean"](data["hyper_mean"])
    hyper_std = approximator.standardize_layers["hyper_std"](data["hyper_std"])
    shared_std = approximator.standardize_layers["shared_std"](data["shared_std"])

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate([hyper_mean, hyper_std, shared_std, data_conditions, expanded])

    assert keras.ops.all(conditions[1] == expected_output)

    # output has the correct shape
    expected_shape = (2, 6, 15)
    assert keras.ops.shape(conditions[1]) == expected_shape


def test_inference_conditions_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        inference_conditions_by_network,
        summary_input,
        summary_outputs_by_network,
    )

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data)

    # output has the correct shape
    expected_shape = (2, 33)
    assert keras.ops.shape(conditions[0]) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[2]

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate([data_conditions, expanded])

    assert keras.ops.all(conditions[0] == expected_output)

    # output has the correct shape
    expected_shape = (2, data.meta["N_classrooms"], 26)
    assert keras.ops.shape(conditions[1]) == expected_shape

    # output is as expected
    data_conditions = summary_outputs[1]
    school_mu = approximator.standardize_layers["school_mu"](data["school_mu"])
    school_sigma = approximator.standardize_layers["school_sigma"](data["school_sigma"])
    shared_sigma = approximator.standardize_layers["shared_sigma"](data["shared_sigma"])

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate([school_mu, school_sigma, shared_sigma, data_conditions, expanded])

    assert keras.ops.all(conditions[1] == expected_output)

    # output has the correct shape
    expected_shape = (2, data.meta["N_classrooms"], data.meta["N_students"], 18)
    assert keras.ops.shape(conditions[2]) == expected_shape

    # output is as expected
    data_conditions = summary_outputs[0]
    school_mu = approximator.standardize_layers["school_mu"](data["school_mu"])
    school_sigma = approximator.standardize_layers["school_sigma"](data["school_sigma"])
    classroom_mu = approximator.standardize_layers["classroom_mu"](data["classroom_mu"])
    classroom_sigma = approximator.standardize_layers["classroom_sigma"](data["classroom_sigma"])
    shared_sigma = approximator.standardize_layers["shared_sigma"](data["shared_sigma"])

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate(
        [school_mu, school_sigma, classroom_mu, classroom_sigma, shared_sigma, data_conditions, expanded]
    )

    assert keras.ops.all(conditions[2] == expected_output)


def test_inference_conditions_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.network_assignment import (
        data_conditions_by_network,
        inference_conditions_by_network,
        summary_input,
        summary_outputs_by_network,
    )

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data)

    # output has the correct shape
    expected_shape = (2, 22)
    assert keras.ops.shape(conditions[0]) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[1]

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate([data_conditions, expanded])

    assert keras.ops.all(conditions[0] == expected_output)

    # output has the correct shape
    expected_shape = (2, data.meta["num_questions"], 16)
    assert keras.ops.shape(conditions[1]) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[0]

    mu_question_mean = approximator.standardize_layers["mu_question_mean"](data["mu_question_mean"])
    sigma_question_mean = approximator.standardize_layers["sigma_question_mean"](data["sigma_question_mean"])
    mu_question_std = approximator.standardize_layers["mu_question_std"](data["mu_question_std"])
    sigma_question_std = approximator.standardize_layers["sigma_question_std"](data["sigma_question_std"])

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate(
        [mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std, data_conditions, expanded]
    )

    assert keras.ops.all(conditions[1] == expected_output)

    # output has the correct shape
    expected_shape = (
        2,
        data.meta["num_students"],
        30 + 40 + 4 + 2,
    )
    assert keras.ops.shape(conditions[2]) == expected_shape

    # output is as expected
    data_conditions = data_conditions_by_network(approximator, data)
    additional_conditions = summary_outputs[3]

    mu_question_mean = approximator.standardize_layers["mu_question_mean"](data["mu_question_mean"])
    sigma_question_mean = approximator.standardize_layers["sigma_question_mean"](data["sigma_question_mean"])
    mu_question_std = approximator.standardize_layers["mu_question_std"](data["mu_question_std"])
    sigma_question_std = approximator.standardize_layers["sigma_question_std"](data["sigma_question_std"])

    node_reps = summary_input(approximator, data).shape[1:-1]
    squared = keras.ops.sqrt(node_reps)
    expanded = keras.ops.expand_dims(squared, axis=0)
    expected_output = concatenate(
        [
            mu_question_mean,
            sigma_question_mean,
            mu_question_std,
            sigma_question_std,
            data_conditions[2],
            additional_conditions,
            expanded,
        ]
    )

    assert keras.ops.all(conditions[2] == expected_output)
