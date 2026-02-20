import pytest


def test_data_condition_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = single_level_simulator.sample(1)
    assert data_condition_shapes_by_network(single_level_approximator) == {0: ("B", 10)}
    assert data_condition_shapes_by_network(single_level_approximator, data.meta) == {0: ("B", 10)}


def test_data_condition_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = two_level_simulator.sample(1)
    assert data_condition_shapes_by_network(two_level_approximator) == {0: ("B", 20), 1: ("B", 6, 10)}
    assert data_condition_shapes_by_network(two_level_approximator, data.meta) == {0: ("B", 20), 1: ("B", 6, 10)}


def test_data_condition_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = three_level_simulator.sample(1)
    assert data_condition_shapes_by_network(three_level_approximator) == {
        0: ("B", 30),
        1: ("B", "N_classrooms", 20),
        2: ("B", "N_classrooms", "N_students", 10),
    }
    assert data_condition_shapes_by_network(three_level_approximator, data.meta) == {
        0: ("B", 30),
        1: ("B", data.meta["N_classrooms"], 20),
        2: ("B", data.meta["N_classrooms"], data.meta["N_students"], 10),
    }


def test_data_condition_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.shape_inference import data_condition_shapes_by_network

    data = crossed_design_irt_simulator.sample(1)
    assert data_condition_shapes_by_network(crossed_design_irt_approximator) == {
        0: ("B", 20),
        1: ("B", "num_questions", 10),
        2: ("B", 20),
    }
    assert data_condition_shapes_by_network(crossed_design_irt_approximator, data.meta) == {
        0: ("B", 20),
        1: ("B", data.meta["num_questions"], 10),
        2: ("B", 20),
    }


def test_summary_output_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = single_level_simulator.sample(1)
    assert summary_output_shapes_by_network(single_level_approximator) == {0: ("B", 10)}
    assert summary_output_shapes_by_network(single_level_approximator, {**data.meta, "B": 1}) == {0: (1, 10)}


def test_summary_output_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = two_level_simulator.sample(1)
    assert summary_output_shapes_by_network(two_level_approximator) == {0: ("B", 6, 10), 1: ("B", 20)}
    assert summary_output_shapes_by_network(two_level_approximator, {**data.meta, "B": 1}) == {
        0: (1, 6, 10),
        1: (1, 20),
    }


def test_summary_output_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = three_level_simulator.sample(1)
    assert summary_output_shapes_by_network(three_level_approximator) == {
        0: ("B", "N_classrooms", "N_students", 10),
        1: ("B", "N_classrooms", 20),
        2: ("B", 30),
    }
    assert summary_output_shapes_by_network(three_level_approximator, {**data.meta, "B": 1}) == {
        0: (1, data.meta["N_classrooms"], data.meta["N_students"], 10),
        1: (1, data.meta["N_classrooms"], 20),
        2: (1, 30),
    }


def test_summary_output_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_output_shapes_by_network

    data = crossed_design_irt_simulator.sample(1)
    assert summary_output_shapes_by_network(crossed_design_irt_approximator) == {
        0: ("B", "num_questions", 10),
        1: ("B", 20),
    }
    assert summary_output_shapes_by_network(crossed_design_irt_approximator, {**data.meta, "B": 1}) == {
        0: (1, data.meta["num_questions"], 10),
        1: (1, 20),
    }


def test_summary_input__shape_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = single_level_simulator.sample(1)
    assert summary_input_shape(single_level_approximator, data.meta) == ("B", data.meta["N"], 2)


def test_summary_input_shape_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = two_level_simulator.sample(1)
    assert summary_input_shape(two_level_approximator) == ("B", 6, 10, 1)
    assert summary_input_shape(two_level_approximator, {**data.meta, "B": 1}) == (1, 6, 10, 1)


def test_summary_input_shape_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = three_level_simulator.sample(1)
    assert summary_input_shape(three_level_approximator) == ("B", "N_classrooms", "N_students", "N_scores", 1)
    assert (
        summary_input_shape(three_level_approximator, {**data.meta, "B": 1})
        == three_level_approximator._data_shapes(data)["y"]
    )


def test_summary_input_shape_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import summary_input_shape

    data = crossed_design_irt_simulator.sample(1)
    assert summary_input_shape(crossed_design_irt_approximator) == ("B", "num_questions", "num_students", 1)
    assert (
        summary_input_shape(crossed_design_irt_approximator, {**data.meta, "B": 1})
        == crossed_design_irt_approximator._data_shapes(data)["obs"]
    )


def test_inference_variable_shapes_by_network_single_level(single_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: ("B", 3),  # 3 variables (beta (2-dimensional), sigma)
    }
    assert inference_variable_shapes_by_network(single_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_two_level(two_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: ("B", 3),  # 3 variables (hyper_mean, hyper_std, shared_std)
        1: ("B", 6, 1),  # 1 variable (local_mean)
    }
    assert inference_variable_shapes_by_network(two_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_three_level(three_level_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: ("B", 3),  # 3 variables (school_mu, school_sigma, shared_sigma)
        1: ("B", "N_classrooms", 2),  # 2 variables (classroom_mu, classroom_sigma)
        2: ("B", "N_classrooms", "N_students", 2),  # 2 variables (student_mu, student_sigma)
    }

    assert inference_variable_shapes_by_network(three_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_crossed_design_irt(crossed_design_irt_approximator):
    from bayesflow.experimental.graphical_approximator.shape_inference import inference_variable_shapes_by_network

    expected_shapes = {
        0: ("B", 4),  # 4 variables (mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std)
        1: ("B", "num_questions", 3),  # 3 variables (question_mean, question_std, question_difficulty)
        2: ("B", "num_students", 1),  # 1 variable (student_ability)
    }

    assert inference_variable_shapes_by_network(crossed_design_irt_approximator) == expected_shapes


def test_concatenate_shapes():
    from bayesflow.experimental.graphical_approximator.shape_inference import concatenate_shapes

    assert concatenate_shapes([(2, 3), (2, 3), (2, 10, 5)]) == (2, 10, 11)
    assert concatenate_shapes([("B", 1), ("B", "N", 3), ("B", "N", 2)]) == ("B", "N", 6)


def test_replace_placeholders():
    from bayesflow.experimental.graphical_approximator.shape_inference import replace_placeholders

    assert replace_placeholders(("B", "N", 3), {"B": 4, "N": 3}) == (4, 3, 3)
    assert replace_placeholders((1, 2, 3), {"B": 4, "N": 3}) == (1, 2, 3)


def test_expand_shape_rank():
    from bayesflow.experimental.graphical_approximator.shape_inference import expand_shape_rank

    assert expand_shape_rank((2, 1, 20), 5) == (2, 1, 1, 1, 20)
    assert expand_shape_rank(("B", 2), 4) == ("B", 1, 1, 2)
    with pytest.raises(ValueError):
        expand_shape_rank((1, 2, 3, 4), 3)


def test_stack_shapes():
    from bayesflow.experimental.graphical_approximator.shape_inference import stack_shapes

    assert stack_shapes((2, 20), (2, 3, 3)) == (2, 3, 23)
    assert stack_shapes(("B", 3), ("B", 1)) == ("B", 4)
    with pytest.raises(ValueError):
        stack_shapes(("B", 3), (2, 3))
