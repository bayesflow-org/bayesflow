import sympy as sp


def test_data_condition_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = single_level_simulator.sample(1)
    data_shapes = single_level_approximator._data_shapes(data)
    assert data_condition_shapes_by_network(single_level_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("summary_dim_0"))
    }
    assert data_condition_shapes_by_network(single_level_approximator, data_shapes) == {0: (1, 10)}


def test_data_condition_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = two_level_simulator.sample(1)
    data_shapes = two_level_approximator._data_shapes(data)
    assert data_condition_shapes_by_network(two_level_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("summary_dim_1")),
        1: (sp.Symbol("B"), 6, sp.Symbol("summary_dim_0")),
    }
    assert data_condition_shapes_by_network(two_level_approximator, data_shapes) == {0: (1, 20), 1: (1, 6, 10)}


def test_data_condition_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = three_level_simulator.sample(1)
    data_shapes = three_level_approximator._data_shapes(data)
    assert data_condition_shapes_by_network(three_level_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("summary_dim_2")),
        1: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("summary_dim_1")),
        2: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("N_students"), sp.Symbol("summary_dim_0")),
    }
    assert data_condition_shapes_by_network(three_level_approximator, data_shapes) == {
        0: (1, 30),
        1: (1, data.meta["N_classrooms"], 20),
        2: (1, data.meta["N_classrooms"], data.meta["N_students"], 10),
    }


def test_data_condition_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = crossed_design_irt_simulator.sample(1)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    assert data_condition_shapes_by_network(crossed_design_irt_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("summary_dim_1")),
        1: (sp.Symbol("B"), sp.Symbol("num_questions"), sp.Symbol("summary_dim_0")),
        2: (sp.Symbol("B"), sp.Symbol("num_students"), sp.Symbol("summary_dim_2")),
    }
    assert data_condition_shapes_by_network(crossed_design_irt_approximator, data_shapes) == {
        0: (1, 20),
        1: (1, data.meta["num_questions"], 10),
        2: (1, data.meta["num_students"], 30),
    }


def test_summary_output_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = single_level_simulator.sample(1)
    data_shapes = single_level_approximator._data_shapes(data)
    assert summary_output_shapes_by_network(single_level_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("summary_dim_0"))
    }
    assert summary_output_shapes_by_network(single_level_approximator, data_shapes) == {0: (1, 10)}


def test_summary_output_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = two_level_simulator.sample(1)
    data_shapes = two_level_approximator._data_shapes(data)
    assert summary_output_shapes_by_network(two_level_approximator) == {
        0: (sp.Symbol("B"), 6, sp.Symbol("summary_dim_0")),
        1: (sp.Symbol("B"), sp.Symbol("summary_dim_1")),
    }
    assert summary_output_shapes_by_network(two_level_approximator, data_shapes) == {
        0: (1, 6, 10),
        1: (1, 20),
    }


def test_summary_output_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = three_level_simulator.sample(1)
    data_shapes = three_level_approximator._data_shapes(data)
    assert summary_output_shapes_by_network(three_level_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("N_students"), sp.Symbol("summary_dim_0")),
        1: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("summary_dim_1")),
        2: (sp.Symbol("B"), sp.Symbol("summary_dim_2")),
    }
    assert summary_output_shapes_by_network(three_level_approximator, data_shapes) == {
        0: (1, data.meta["N_classrooms"], data.meta["N_students"], 10),
        1: (1, data.meta["N_classrooms"], 20),
        2: (1, 30),
    }


def test_summary_output_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = crossed_design_irt_simulator.sample(1)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    assert summary_output_shapes_by_network(crossed_design_irt_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("num_questions"), sp.Symbol("summary_dim_0")),
        1: (sp.Symbol("B"), sp.Symbol("summary_dim_1")),
        2: (sp.Symbol("B"), sp.Symbol("num_students"), sp.Symbol("summary_dim_2")),
        3: (sp.Symbol("B"), sp.Symbol("summary_dim_3")),
    }
    assert summary_output_shapes_by_network(crossed_design_irt_approximator, data_shapes) == {
        0: (1, data.meta["num_questions"], 10),
        1: (1, 20),
        2: (1, data.meta["num_students"], 30),
        3: (1, 40),
    }


def test_summary_input_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = single_level_simulator.sample(1)
    data_shapes = single_level_approximator._data_shapes(data)
    assert summary_input_shapes_by_network(single_level_approximator, data_shapes) == {0: (1, data.meta["N"], 2)}


def test_summary_input_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = two_level_simulator.sample(1)
    data_shapes = two_level_approximator._data_shapes(data)
    assert summary_input_shapes_by_network(two_level_approximator) == {
        0: (sp.Symbol("B"), 6, 10, 1),
        1: (sp.Symbol("B"), 6, sp.Symbol("summary_dim_0")),
    }
    assert summary_input_shapes_by_network(two_level_approximator, data_shapes) == {
        0: (1, 6, 10, 1),
        1: (1, 6, 10),
    }


def test_summary_input_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = three_level_simulator.sample(1)
    data_shapes = three_level_approximator._data_shapes(data)
    assert summary_input_shapes_by_network(three_level_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("N_students"), sp.Symbol("N_scores"), 1),
        1: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("N_students"), sp.Symbol("summary_dim_0")),
        2: (sp.Symbol("B"), sp.Symbol("N_classrooms"), sp.Symbol("summary_dim_1")),
    }
    assert summary_input_shapes_by_network(three_level_approximator, data_shapes) == {
        0: (1, data.meta["N_classrooms"], data.meta["N_students"], data.meta["N_scores"], 1),
        1: (1, data.meta["N_classrooms"], data.meta["N_students"], 10),
        2: (1, data.meta["N_classrooms"], 20),
    }


def test_summary_input_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shapes_by_network

    data = crossed_design_irt_simulator.sample(1)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    assert summary_input_shapes_by_network(crossed_design_irt_approximator) == {
        0: (sp.Symbol("B"), sp.Symbol("num_questions"), sp.Symbol("num_students"), 1),
        1: (sp.Symbol("B"), sp.Symbol("num_questions"), sp.Symbol("summary_dim_0")),
        2: (sp.Symbol("B"), sp.Symbol("num_students"), sp.Symbol("num_questions"), 1),
        3: (sp.Symbol("B"), sp.Symbol("num_questions"), 3),
    }
    assert summary_input_shapes_by_network(crossed_design_irt_approximator, data_shapes) == {
        0: (1, data.meta["num_questions"], data.meta["num_students"], 1),
        1: (1, data.meta["num_questions"], 10),
        2: (1, data.meta["num_students"], data.meta["num_questions"], 1),
        3: (1, data.meta["num_questions"], 3),
    }


def test_summary_input_shape_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = single_level_simulator.sample(1)
    data_shapes = single_level_approximator._data_shapes(data)
    assert summary_input_shape(single_level_approximator, data_shapes) == (1, data.meta["N"], 2)


def test_summary_input_shape_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = two_level_simulator.sample(1)
    data_shapes = two_level_approximator._data_shapes(data)
    assert summary_input_shape(two_level_approximator) == (sp.Symbol("B"), 6, 10, 1)
    assert summary_input_shape(two_level_approximator, data_shapes) == (1, 6, 10, 1)


def test_summary_input_shape_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = three_level_simulator.sample(1)
    data_shapes = three_level_approximator._data_shapes(data)
    assert summary_input_shape(three_level_approximator) == (
        sp.Symbol("B"),
        sp.Symbol("N_classrooms"),
        sp.Symbol("N_students"),
        sp.Symbol("N_scores"),
        1,
    )
    assert (
        summary_input_shape(three_level_approximator, data_shapes) == three_level_approximator._data_shapes(data)["y"]
    )


def test_summary_input_shape_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = crossed_design_irt_simulator.sample(1)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    assert summary_input_shape(crossed_design_irt_approximator) == (
        sp.Symbol("B"),
        sp.Symbol("num_questions"),
        sp.Symbol("num_students"),
        1,
    )
    assert (
        summary_input_shape(crossed_design_irt_approximator, data_shapes)
        == crossed_design_irt_approximator._data_shapes(data)["obs"]
    )


def test_inference_variable_shapes_by_network_single_level(single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: (sp.Symbol("B"), 3),  # 3 variables (beta (2-dimensional), sigma)
    }
    assert inference_variable_shapes_by_network(single_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_two_level(two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: (sp.Symbol("B"), 3),  # 3 variables (hyper_mean, hyper_std, shared_std)
        1: (sp.Symbol("B"), 6, 1),  # 1 variable (local_mean)
    }
    assert inference_variable_shapes_by_network(two_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_three_level(three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: (sp.Symbol("B"), 3),  # 3 variables (school_mu, school_sigma, shared_sigma)
        1: (sp.Symbol("B"), sp.Symbol("N_classrooms"), 2),  # 2 variables (classroom_mu, classroom_sigma)
        2: (
            sp.Symbol("B"),
            sp.Symbol("N_classrooms"),
            sp.Symbol("N_students"),
            2,
        ),  # 2 variables (student_mu, student_sigma)
    }

    assert inference_variable_shapes_by_network(three_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_crossed_design_irt(crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: (
            sp.Symbol("B"),
            4,
        ),  # 4 variables (mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std)
        1: (
            sp.Symbol("B"),
            sp.Symbol("num_questions"),
            3,
        ),  # 3 variables (question_mean, question_std, question_difficulty)
        2: (sp.Symbol("B"), sp.Symbol("num_students"), 1),  # 1 variable (student_ability)
    }

    assert inference_variable_shapes_by_network(crossed_design_irt_approximator) == expected_shapes


def test_inference_condition_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_condition_shapes_by_network

    data_shapes = single_level_approximator._data_shapes(single_level_simulator.sample(2))
    expected_shapes = {0: (2, 11)}  # 10 summary dimensions + 1 node repetition

    assert inference_condition_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_condition_shapes_by_network

    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 22),  # 20 summary dimensions + 0 variables + 2 node reps
        1: (2, 6, 15),  # 10 summary dimensons + 3 variable + 2 node reps
    }

    assert inference_condition_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_condition_shapes_by_network

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 33),  # 30 summary dimensions + 0 variables + 3 node reps
        1: (2, data.meta["N_classrooms"], 26),  # 20 summary dimensions + 3 variables + 3 node reps
        2: (
            2,
            data.meta["N_classrooms"],
            data.meta["N_students"],
            18,
        ),  # 10 summary dimensions + 5 variables + 3 node reps
    }

    assert inference_condition_shapes_by_network(three_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_condition_shapes_by_network

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 22),  # 20 summary dimensions + 0 variables + 2 node reps
        1: (2, data.meta["num_questions"], 16),  # 10 summary dimensions + 4 variables + 2 node reps
        2: (
            2,
            data.meta["num_students"],
            30 + 40 + 4 + 2,
        ),  # 20 summary dimensions + num_questions * 3 + 4 variables + 2 node reps
    }

    assert inference_condition_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes
