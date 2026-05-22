import keras


def test_inference_conditions_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_conditions_by_network

    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data, approximator.summary_registry)

    expected_shape = (2, 11)  # 10 summary dimensions + 1 node repetition
    assert keras.ops.shape(conditions[0]) == expected_shape


def test_inference_conditions_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_conditions_by_network

    data = two_level_simulator.sample(2)
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data, approximator.summary_registry)

    expected_shape = (2, 22)  # 20 summary dimensions + 2 node reps
    assert keras.ops.shape(conditions[0]) == expected_shape

    expected_shape = (2, 6, 35)  # 30 summary dimensions + 3 variables + 2 node reps
    assert keras.ops.shape(conditions[1]) == expected_shape


def test_inference_conditions_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_conditions_by_network

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data, approximator.summary_registry)

    expected_shape = (2, 33)  # 30 summary dimensions + 3 node reps
    assert keras.ops.shape(conditions[0]) == expected_shape

    expected_shape = (2, data.meta["N_classrooms"], 56)  # 50 summary dimensions + 3 variables + 3 node reps
    assert keras.ops.shape(conditions[1]) == expected_shape

    expected_shape = (
        2,
        data.meta["N_classrooms"],
        data.meta["N_students"],
        68,
    )  # 60 summary dimensions + 5 variables + 3 node reps
    assert keras.ops.shape(conditions[2]) == expected_shape


def test_inference_conditions_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_conditions_by_network

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    conditions = inference_conditions_by_network(approximator, data, approximator.summary_registry)

    expected_shape = (2, 22)  # 20 summary dimensions + 2 node reps
    assert keras.ops.shape(conditions[0]) == expected_shape

    expected_shape = (2, data.meta["num_questions"], 16)  # 10 summary dimensions + 4 variables + 2 node reps
    assert keras.ops.shape(conditions[1]) == expected_shape

    expected_shape = (
        2,
        data.meta["num_students"],
        30 + 40 + 4 + 2,
    )  # 30 + 40 summary dimensions + 4 variables + 2 node reps
    assert keras.ops.shape(conditions[2]) == expected_shape


def test_inference_condition_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_condition_shapes_by_network

    data_shapes = single_level_approximator._data_shapes(single_level_simulator.sample(2))
    expected_shapes = {0: (2, 11)}  # 10 summary dimensions + 1 node repetition

    assert inference_condition_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_condition_shapes_by_network

    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 22),  # 20 summary dimensions + 2 node reps
        1: (2, 6, 35),  # 30 summary dimensions + 3 variables + 2 node reps
    }

    assert inference_condition_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_condition_shapes_by_network

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 33),  # 30 summary dimensions + 3 node reps
        1: (2, data.meta["N_classrooms"], 56),  # 50 summary dimensions + 3 variables + 3 node reps
        2: (
            2,
            data.meta["N_classrooms"],
            data.meta["N_students"],
            68,
        ),  # 60 summary dimensions + 5 variables + 3 node reps
    }

    assert inference_condition_shapes_by_network(three_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_condition_shapes_by_network

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 22),  # 20 summary dimensions + 2 node reps
        1: (2, data.meta["num_questions"], 16),  # 10 summary dimensions + 4 variables + 2 node reps
        2: (
            2,
            data.meta["num_students"],
            76,
        ),  # 30 + 40 summary dimensions + 4 variables + 2 node reps
    }

    assert inference_condition_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_no_data_shapes(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_conditions import inference_condition_shapes_by_network

    result = inference_condition_shapes_by_network(single_level_approximator)
    assert 0 in result
