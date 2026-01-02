from bayesflow.experimental.graphical_approximator.utils import (
    expand_shape_rank,
    stack_shapes,
    concatenate_shapes,
    add_sample_dimension,
    concatenate,
    repetitions_from_data_shape,
    inference_condition_shapes_by_network,
    inference_variable_shapes_by_network,
    data_condition_shapes_by_network,
    summary_input_shapes_by_network,
    summary_output_shapes_by_network,
    summary_input_shape,
    add_node_reps_to_conditions,
    prepare_inference_conditions,
    summary_outputs_by_network,
)
import keras


def test_expand_shape_rank():
    x = (10, 2, 3)

    assert expand_shape_rank(x, 4) == (10, 2, 1, 3)
    assert expand_shape_rank(x, 5) == (10, 2, 1, 1, 3)


def test_stack_shapes():
    a = (10, 2, 3)
    b = (32, 1)

    assert stack_shapes(a, b, axis=-1) == (32, 2, 4)
    assert stack_shapes(a, b, axis=0) == (42, 2, 3)
    assert stack_shapes(a, b, axis=1) == (32, 3, 3)


def test_concatenate_shapes():
    assert concatenate_shapes([(7, 5, 2), (3, 20)]) == (7, 5, 22)
    assert concatenate_shapes([(3, 20), (7, 5, 2)]) == (7, 5, 22)


def test_add_sample_dimension():
    x = keras.random.normal((10, 5))
    y = add_sample_dimension(x, 55)
    assert keras.ops.shape(y) == (10, 55, 5)

    x = keras.random.normal((10,))
    y = add_sample_dimension(x, 55)
    assert keras.ops.shape(y) == (10, 55)


def test_concatenate():
    x = keras.random.normal((20, 5))
    y = keras.random.normal((20, 15, 3))
    z = concatenate([x, y])
    assert keras.ops.shape(z) == (20, 15, 8)

    x = keras.random.normal((20, 1))
    y = keras.random.normal((20, 15, 3))
    z = concatenate([x, y])
    assert keras.ops.shape(z) == (20, 15, 4)


def test_repetitions_from_data_shape(single_level_simulator, single_level_approximator):
    data = single_level_simulator.sample(10)
    data_shapes = single_level_approximator._data_shapes(data)

    assert repetitions_from_data_shape(single_level_approximator, data_shapes) == {}


def test_repetitions_from_data_shape_two_level(two_level_simulator, two_level_approximator):
    data = two_level_simulator.sample(10)
    data_shapes = two_level_approximator._data_shapes(data)

    assert repetitions_from_data_shape(two_level_approximator, data_shapes) == {"locals": 6, "y": 10}


def test_repetitions_from_data_shape_three_level(three_level_simulator, three_level_approximator):
    data = three_level_simulator.sample(10)
    data_shapes = three_level_approximator._data_shapes(data)

    assert repetitions_from_data_shape(three_level_approximator, data_shapes) == {
        "classrooms": data.meta["N_classrooms"],
        "scores": data.meta["N_scores"],
        "students": data.meta["N_students"],
    }


def test_repetitions_from_data_shape_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    data = crossed_design_irt_simulator.sample(10)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    assert repetitions_from_data_shape(crossed_design_irt_approximator, data_shapes) == {
        "questions": data.meta["num_questions"],
        "students": data.meta["num_students"],
    }


def test_inference_condition_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    # network composition:
    # {0: ["prior"]}
    # network conditions:
    # {1: ["likelihood"]}
    # data shapes:
    # {
    #   "beta": (2, 2),
    #   "sigma": (2, 1),
    #   "x": (2, 12, 1),
    #   "y": (2, 12, 1),
    # }
    # summary output shapes:
    # {0: (2, 10)}
    data_shapes = single_level_approximator._data_shapes(single_level_simulator.sample(2))
    expected_shapes = {0: (2, 10)}  # 10 summary dimensions

    assert inference_condition_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    # network composition:
    # {0: ["hypers", "shared"], 1: ["locals"]}
    # network conditions:
    # {0: ["y"], 1: ["hypers", "shared", "y"]}
    # data shapes:
    # {
    #   'hyper_mean': (2, 1),
    #   'hyper_std': (2, 1),
    #   'local_mean': (2, 6, 1),
    #   'shared_std': (2, 1),
    #   'y': (2, 6, 10, 1)
    # }
    # summary output shapes:
    # {0: (2, 6, 10), 1: (2, 20)}
    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 22),  # 20 summary dimensions + 0 variables + 2 node reps
        1: (2, 6, 15),  # 10 summary dimensons + 3 variable + 2 node reps
    }

    assert inference_condition_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_inference_condition_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    # network composition:
    # {0: ["schools, "shared"], 1: ["classrooms"], 2: ["students"]}
    # network conditions:
    # {0: ["scores"], 1: ["schools", "shared", "scores"], 2: ["schools", "classrooms", "shared", "scores"]}
    # data shapes:
    # {
    #   'school_mu': (2, 1),
    #   'school_sigma': (2, 1),
    #   'classroom_mu': (2, N_classrooms, 1),
    #   'classroom_sigma': (2, N_classrooms, 1),
    #   'shared_sigma': (2, 1),
    #   'student_mu': (2, N_classrooms, N_students, 1),
    #   'student_sigma': (2, N_classrooms, N_students, 1),
    #   'y': (2, N_classrooms, N_students, N_scores, 1)
    # }
    # summary output shapes:
    # {0: (2, N_classrooms, N_students, 10), 1: (2, N_classrooms, 20), 2: (2, 30)}
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
    # network composition:
    # {0: ["schools"], 1: ["questions"], 2: ["students"]}
    # network conditions:
    # {0: ["observations"], 1: ["schools", "observations"], 2: ["schools", "questions", "observations"]}
    # data shapes:
    # {
    #   'mu_question_mean': (2, 1),
    #   'sigma_question_mean': (2, 1),
    #   'mu_question_std': (2, 1),
    #   'sigma_question_std': (2, 1),
    #   'question_mean': (2, num_questions, 1),
    #   'question_std': (2, num_questions, 1),
    #   'question_difficulty': (2, num_questions, 1),
    #   'student_ability': (2, num_students, 1),
    #   'obs': (2, num_questions, num_students, 1)
    # }
    # summary output shapes:
    # {0: (2, num_students, 10), 1: (2, 20)}
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 22),  # 20 summary dimensions + 0 variables + 2 node reps
        1: (2, 26),  # 20 summary dimensions + 4 variables + 2 node reps
        2: (
            2,
            data.meta["num_students"],
            10 + data.meta["num_questions"] * 3 + 4 + 2,
        ),  # 10 summary dimensions + num_questions * 3 + 4 variables + 2 node reps
    }

    assert inference_condition_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes


def test_inference_variable_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    # network composition:
    # {0: ["prior"]}
    # network conditions:
    # {1: ["likelihood"]}
    # data shapes:
    # {
    #   "beta": (2, 2),
    #   "sigma": (2, 1),
    #   "x": (2, 12, 1),
    #   "y": (2, 12, 1),
    # }
    data_shapes = single_level_approximator._data_shapes(single_level_simulator.sample(2))
    expected_shapes = {0: (2, 3)}  # 3 variables (beta, sigma)

    assert inference_variable_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_inference_variable_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    # network composition:
    # {0: ["hypers", "shared"], 1: ["locals"]}
    # network conditions:
    # {0: ["y"], 1: ["hypers", "shared", "y"]}
    # data shapes:
    # {
    #   'hyper_mean': (2, 1),
    #   'hyper_std': (2, 1),
    #   'local_mean': (2, 6, 1),
    #   'shared_std': (2, 1),
    #   'y': (2, 6, 10, 1)
    # }
    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 3),  # 3 variables (hyper_mean, hyper_std, shared_std)
        1: (2, 6, 1),  # 1 variable (local_mean)
    }

    assert inference_variable_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_inference_variable_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    # network composition:
    # {0: ["schools, "shared"], 1: ["classrooms"], 2: ["students"]}
    # network conditions:
    # {0: ["scores"], 1: ["schools", "shared", "scores"], 2: ["schools", "classrooms", "shared", "scores"]}
    # data shapes:
    # {
    #   'school_mu': (2, 1),
    #   'school_sigma': (2, 1),
    #   'classroom_mu': (2, N_classrooms, 1),
    #   'classroom_sigma': (2, N_classrooms, 1),
    #   'shared_sigma': (2, 1),
    #   'student_mu': (2, N_classrooms, N_students, 1),
    #   'student_sigma': (2, N_classrooms, N_students, 1),
    #   'y': (2, N_classrooms, N_students, N_scores, 1)
    # }
    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 3),  # 3 variables (school_mu, school_sigma, shared_sigma)
        1: (2, data.meta["N_classrooms"], 2),  # 2 variables (classroom_mu, classroom_sigma)
        2: (2, data.meta["N_classrooms"], data.meta["N_students"], 2),  # 2 variables (student_mu, student_sigma)
    }

    assert inference_variable_shapes_by_network(three_level_approximator, data_shapes) == expected_shapes


def test_inference_variable_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    # network composition:
    # {0: ["schools"], 1: ["questions"], 2: ["students"]}
    # network conditions:
    # {0: ["observations"], 1: ["schools", "observations"], 2: ["schools", "questions", "observations"]}
    # data shapes:
    # {
    #   'mu_question_mean': (2, 1),
    #   'sigma_question_mean': (2, 1),
    #   'mu_question_std': (2, 1),
    #   'sigma_question_std': (2, 1),
    #   'question_mean': (2, num_questions, 1),
    #   'question_std': (2, num_questions, 1),
    #   'question_difficulty': (2, num_questions, 1),
    #   'student_ability': (2, num_students, 1),
    #   'obs': (2, num_questions, num_students, 1)
    # }
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 4),  # 4 variables (mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std)
        1: (2, data.meta["num_questions"] * 3),  # 3 variables (question_mean, question_std, question_difficulty)
        2: (2, data.meta["num_students"], 1),  # 1 variable (student_ability)
    }

    assert inference_variable_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes


def test_data_condition_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    # network composition:
    # {0: ["prior"]}
    # network conditions:
    # {1: ["likelihood"]}
    # data shapes:
    # {
    #   "beta": (2, 2),
    #   "sigma": (2, 1),
    #   "x": (2, 12, 1),
    #   "y": (2, 12, 1),
    # }
    # summary output shapes:
    # {0: (2, 10)}
    data_shapes = single_level_approximator._data_shapes(single_level_simulator.sample(2))
    expected_shapes = {0: (2, 10)}

    assert data_condition_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_data_condition_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    # network composition:
    # {0: ["hypers", "shared"], 1: ["locals"]}
    # network conditions:
    # {0: ["y"], 1: ["hypers", "shared", "y"]}
    # data shapes:
    # {
    #   'hyper_mean': (2, 1),
    #   'hyper_std': (2, 1),
    #   'local_mean': (2, 6, 1),
    #   'shared_std': (2, 1),
    #   'y': (2, 6, 10, 1)
    # }
    # summary output shapes:
    # {0: (2, 6, 10), 1: (2, 20)}
    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 20),  # 20 summary dimensions
        1: (2, 6, 10),  # 10 summary dimensons
    }

    assert data_condition_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_data_condition_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    # network composition:
    # {0: ["schools, "shared"], 1: ["classrooms"], 2: ["students"]}
    # network conditions:
    # {0: ["scores"], 1: ["schools", "shared", "scores"], 2: ["schools", "classrooms", "shared", "scores"]}
    # data shapes:
    # {
    #   'school_mu': (2, 1),
    #   'school_sigma': (2, 1),
    #   'classroom_mu': (2, N_classrooms, 1),
    #   'classroom_sigma': (2, N_classrooms, 1),
    #   'shared_sigma': (2, 1),
    #   'student_mu': (2, N_classrooms, N_students, 1),
    #   'student_sigma': (2, N_classrooms, N_students, 1),
    #   'y': (2, N_classrooms, N_students, N_scores, 1)
    # }
    # summary output shapes:
    # {0: (2, N_classrooms, N_students, 10), 1: (2, N_classrooms, 20), 2: (2, 30)}
    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 30),  # 30 summary dimensions
        1: (2, data.meta["N_classrooms"], 20),  # 20 summary dimensions
        2: (
            2,
            data.meta["N_classrooms"],
            data.meta["N_students"],
            10,
        ),  # 10 summary dimensions
    }

    assert data_condition_shapes_by_network(three_level_approximator, data_shapes) == expected_shapes


def test_data_condition_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    # network composition:
    # {0: ["schools"], 1: ["questions"], 2: ["students"]}
    # network conditions:
    # {0: ["observations"], 1: ["schools", "observations"], 2: ["schools", "questions", "observations"]}
    # data shapes:
    # {
    #   'mu_question_mean': (2, 1),
    #   'sigma_question_mean': (2, 1),
    #   'mu_question_std': (2, 1),
    #   'sigma_question_std': (2, 1),
    #   'question_mean': (2, num_questions, 1),
    #   'question_std': (2, num_questions, 1),
    #   'question_difficulty': (2, num_questions, 1),
    #   'student_ability': (2, num_students, 1),
    #   'obs': (2, num_questions, num_students, 1)
    # }
    # summary output shapes:
    # {0: (2, num_students, 10), 1: (2, 20)}
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, 20),  # 20 summary dimensions
        1: (2, 20),  # 20 summary dimensions
        2: (
            2,
            data.meta["num_students"],
            10,
        ),  # 10 summary dimensions
    }

    assert data_condition_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes


def test_summary_input_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    # network composition:
    # {0: ["prior"]}
    # network conditions:
    # {1: ["likelihood"]}
    # data shapes:
    # {
    #   "beta": (2, 2),
    #   "sigma": (2, 1),
    #   "x": (2, N, 1),
    #   "y": (2, N, 1),
    # }
    # summary output shapes:
    # {0: (2, 10)}
    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)
    expected_shapes = {0: (2, data.meta["N"], 2)}

    assert summary_input_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_summary_input_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    # network composition:
    # {0: ["hypers", "shared"], 1: ["locals"]}
    # network conditions:
    # {0: ["y"], 1: ["hypers", "shared", "y"]}
    # data shapes:
    # {
    #   'hyper_mean': (2, 1),
    #   'hyper_std': (2, 1),
    #   'local_mean': (2, 6, 1),
    #   'shared_std': (2, 1),
    #   'y': (2, 6, 10, 1)
    # }
    # summary output shapes:
    # {0: (2, 6, 10), 1: (2, 20)}
    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 6, 10, 1),
        1: (2, 6, 10),
    }

    assert summary_input_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_summary_input_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    # network composition:
    # {0: ["schools, "shared"], 1: ["classrooms"], 2: ["students"]}
    # network conditions:
    # {0: ["scores"], 1: ["schools", "shared", "scores"], 2: ["schools", "classrooms", "shared", "scores"]}
    # data shapes:
    # {
    #   'school_mu': (2, 1),
    #   'school_sigma': (2, 1),
    #   'classroom_mu': (2, N_classrooms, 1),
    #   'classroom_sigma': (2, N_classrooms, 1),
    #   'shared_sigma': (2, 1),
    #   'student_mu': (2, N_classrooms, N_students, 1),
    #   'student_sigma': (2, N_classrooms, N_students, 1),
    #   'y': (2, N_classrooms, N_students, N_scores, 1)
    # }
    # summary output shapes:
    # {0: (2, N_classrooms, N_students, 10), 1: (2, N_classrooms, 20), 2: (2, 30)}
    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, data.meta["N_classrooms"], data.meta["N_students"], data.meta["N_scores"], 1),
        1: (2, data.meta["N_classrooms"], data.meta["N_students"], 10),
        2: (2, data.meta["N_classrooms"], 20),
    }

    assert summary_input_shapes_by_network(three_level_approximator, data_shapes) == expected_shapes


def test_summary_input_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    # network composition:
    # {0: ["schools"], 1: ["questions"], 2: ["students"]}
    # network conditions:
    # {0: ["observations"], 1: ["schools", "observations"], 2: ["schools", "questions", "observations"]}
    # data shapes:
    # {
    #   'mu_question_mean': (2, 1),
    #   'sigma_question_mean': (2, 1),
    #   'mu_question_std': (2, 1),
    #   'sigma_question_std': (2, 1),
    #   'question_mean': (2, num_questions, 1),
    #   'question_std': (2, num_questions, 1),
    #   'question_difficulty': (2, num_questions, 1),
    #   'student_ability': (2, num_students, 1),
    #   'obs': (2, num_questions, num_students, 1)
    # }
    # summary output shapes:
    # {0: (2, num_students, 10), 1: (2, 20)}
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (
            2,
            data.meta["num_students"],
            data.meta["num_questions"],
            1,
        ),  # num_questions and num_students swapped because questions are not amortizable
        1: (2, data.meta["num_students"], 10),
    }

    assert summary_input_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes


def test_summary_output_shapes_by_network_single_level(single_level_simulator, single_level_approximator):
    # network composition:
    # {0: ["prior"]}
    # network conditions:
    # {1: ["likelihood"]}
    # data shapes:
    # {
    #   "beta": (2, 2),
    #   "sigma": (2, 1),
    #   "x": (2, 12, 1),
    #   "y": (2, 12, 1),
    # }
    # summary output shapes:
    # {0: (2, 10)}
    data_shapes = single_level_approximator._data_shapes(single_level_simulator.sample(2))
    expected_shapes = {0: (2, 10)}  # 10 summary dimensions

    assert summary_output_shapes_by_network(single_level_approximator, data_shapes) == expected_shapes


def test_summary_output_shapes_by_network_two_level(two_level_simulator, two_level_approximator):
    # network composition:
    # {0: ["hypers", "shared"], 1: ["locals"]}
    # network conditions:
    # {0: ["y"], 1: ["hypers", "shared", "y"]}
    # data shapes:
    # {
    #   'hyper_mean': (2, 1),
    #   'hyper_std': (2, 1),
    #   'local_mean': (2, 6, 1),
    #   'shared_std': (2, 1),
    #   'y': (2, 6, 10, 1)
    # }
    # summary output shapes:
    # {0: (2, 6, 10), 1: (2, 20)}
    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shapes = {
        0: (2, 6, 10),
        1: (2, 20),
    }

    assert summary_output_shapes_by_network(two_level_approximator, data_shapes) == expected_shapes


def test_summary_output_shapes_by_network_three_level(three_level_simulator, three_level_approximator):
    # network composition:
    # {0: ["schools, "shared"], 1: ["classrooms"], 2: ["students"]}
    # network conditions:
    # {0: ["scores"], 1: ["schools", "shared", "scores"], 2: ["schools", "classrooms", "shared", "scores"]}
    # data shapes:
    # {
    #   'school_mu': (2, 1),
    #   'school_sigma': (2, 1),
    #   'classroom_mu': (2, N_classrooms, 1),
    #   'classroom_sigma': (2, N_classrooms, 1),
    #   'shared_sigma': (2, 1),
    #   'student_mu': (2, N_classrooms, N_students, 1),
    #   'student_sigma': (2, N_classrooms, N_students, 1),
    #   'y': (2, N_classrooms, N_students, N_scores, 1)
    # }
    # summary output shapes:
    # {0: (2, N_classrooms, N_students, 10), 1: (2, N_classrooms, 20), 2: (2, 30)}
    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, data.meta["N_classrooms"], data.meta["N_students"], 10),
        1: (2, data.meta["N_classrooms"], 20),
        2: (2, 30),
    }

    assert summary_output_shapes_by_network(three_level_approximator, data_shapes) == expected_shapes


def test_summary_output_shapes_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    # network composition:
    # {0: ["schools"], 1: ["questions"], 2: ["students"]}
    # network conditions:
    # {0: ["observations"], 1: ["schools", "observations"], 2: ["schools", "questions", "observations"]}
    # data shapes:
    # {
    #   'mu_question_mean': (2, 1),
    #   'sigma_question_mean': (2, 1),
    #   'mu_question_std': (2, 1),
    #   'sigma_question_std': (2, 1),
    #   'question_mean': (2, num_questions, 1),
    #   'question_std': (2, num_questions, 1),
    #   'question_difficulty': (2, num_questions, 1),
    #   'student_ability': (2, num_students, 1),
    #   'obs': (2, num_questions, num_students, 1)
    # }
    # summary output shapes:
    # {0: (2, num_students, 10), 1: (2, 20)}
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shapes = {
        0: (2, data.meta["num_students"], 10),
        1: (2, 20),
    }

    assert summary_output_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes


def test_summary_input_shape_single_level(single_level_simulator, single_level_approximator):
    # network composition:
    # {0: ["prior"]}
    # network conditions:
    # {1: ["likelihood"]}
    # data shapes:
    # {
    #   "beta": (2, 2),
    #   "sigma": (2, 1),
    #   "x": (2, N, 1),
    #   "y": (2, N, 1),
    # }
    # summary output shapes:
    # {0: (2, 10)}
    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)
    expected_shape = (2, data.meta["N"], 2)

    assert summary_input_shape(single_level_approximator, data_shapes) == expected_shape


def test_summary_input_shape_two_level(two_level_simulator, two_level_approximator):
    # network composition:
    # {0: ["hypers", "shared"], 1: ["locals"]}
    # network conditions:
    # {0: ["y"], 1: ["hypers", "shared", "y"]}
    # data shapes:
    # {
    #   'hyper_mean': (2, 1),
    #   'hyper_std': (2, 1),
    #   'local_mean': (2, 6, 1),
    #   'shared_std': (2, 1),
    #   'y': (2, 6, 10, 1)
    # }
    # summary output shapes:
    # {0: (2, 6, 10), 1: (2, 20)}
    data_shapes = two_level_approximator._data_shapes(two_level_simulator.sample(2))
    expected_shape = (2, 6, 10, 1)

    assert summary_input_shape(two_level_approximator, data_shapes) == expected_shape


def test_summary_input_shape_three_level(three_level_simulator, three_level_approximator):
    # network composition:
    # {0: ["schools, "shared"], 1: ["classrooms"], 2: ["students"]}
    # network conditions:
    # {0: ["scores"], 1: ["schools", "shared", "scores"], 2: ["schools", "classrooms", "shared", "scores"]}
    # data shapes:
    # {
    #   'school_mu': (2, 1),
    #   'school_sigma': (2, 1),
    #   'classroom_mu': (2, N_classrooms, 1),
    #   'classroom_sigma': (2, N_classrooms, 1),
    #   'shared_sigma': (2, 1),
    #   'student_mu': (2, N_classrooms, N_students, 1),
    #   'student_sigma': (2, N_classrooms, N_students, 1),
    #   'y': (2, N_classrooms, N_students, N_scores, 1)
    # }
    # summary output shapes:
    # {0: (2, N_classrooms, N_students, 10), 1: (2, N_classrooms, 20), 2: (2, 30)}
    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)
    expected_shape = (2, data.meta["N_classrooms"], data.meta["N_students"], data.meta["N_scores"], 1)

    assert summary_input_shape(three_level_approximator, data_shapes) == expected_shape


def test_summary_input_shape_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    # network composition:
    # {0: ["schools"], 1: ["questions"], 2: ["students"]}
    # network conditions:
    # {0: ["observations"], 1: ["schools", "observations"], 2: ["schools", "questions", "observations"]}
    # data shapes:
    # {
    #   'mu_question_mean': (2, 1),
    #   'sigma_question_mean': (2, 1),
    #   'mu_question_std': (2, 1),
    #   'sigma_question_std': (2, 1),
    #   'question_mean': (2, num_questions, 1),
    #   'question_std': (2, num_questions, 1),
    #   'question_difficulty': (2, num_questions, 1),
    #   'student_ability': (2, num_students, 1),
    #   'obs': (2, num_questions, num_students, 1)
    # }
    # summary output shapes:
    # {0: (2, num_students, 10), 1: (2, 20)}
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)
    expected_shape = (
        2,
        data.meta["num_students"],
        data.meta["num_questions"],
        1,
    )

    assert summary_input_shape(crossed_design_irt_approximator, data_shapes) == expected_shape


def test_add_node_reps_to_conditions():
    x = keras.random.normal((20, 5, 3))
    node_reps = {"a": 20, "b": 30}

    with_reps = add_node_reps_to_conditions(x, node_reps)

    assert keras.ops.all(with_reps[..., :3] == x)
    assert keras.ops.shape(with_reps) == (20, 5, 5)
    assert with_reps[0, 0, 3] == keras.ops.sqrt(20)
    assert with_reps[0, 0, 4] == keras.ops.sqrt(30)


def test_prepare_inference_conditions_single_level(single_level_simulator, single_level_approximator):
    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    conditions = prepare_inference_conditions(approximator, data, 0)

    # output has the correct shape
    expected_shape = (2, 10)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    expected_output = summary_outputs[0]
    assert keras.ops.all(conditions == expected_output)


def test_prepare_inference_conditions_two_level(two_level_simulator, two_level_approximator):
    data = two_level_simulator.sample(2)
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    conditions = prepare_inference_conditions(approximator, data, 0)

    # output has the correct shape
    expected_shape = (2, 22)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[1]
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(data_conditions, repetitions)
    assert keras.ops.all(conditions == expected_output)

    conditions = prepare_inference_conditions(approximator, data, 1)

    # output has the correct shape
    expected_shape = (2, 6, 15)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    data_conditions = summary_outputs[0]
    hyper_mean = approximator.standardize_layers["hyper_mean"](data["hyper_mean"])
    hyper_std = approximator.standardize_layers["hyper_std"](data["hyper_std"])
    shared_std = approximator.standardize_layers["shared_std"](data["shared_std"])
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(
        concatenate([hyper_mean, hyper_std, shared_std, data_conditions]), repetitions
    )

    assert keras.ops.all(conditions == expected_output)


def test_prepare_inference_conditions_three_level(three_level_simulator, three_level_approximator):
    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    conditions = prepare_inference_conditions(approximator, data, 0)

    # output has the correct shape
    expected_shape = (2, 33)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[2]
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(data_conditions, repetitions)
    assert keras.ops.all(conditions == expected_output)

    conditions = prepare_inference_conditions(approximator, data, 1)

    # output has the correct shape
    expected_shape = (2, data.meta["N_classrooms"], 26)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    data_conditions = summary_outputs[1]
    school_mu = approximator.standardize_layers["school_mu"](data["school_mu"])
    school_sigma = approximator.standardize_layers["school_sigma"](data["school_sigma"])
    shared_sigma = approximator.standardize_layers["shared_sigma"](data["shared_sigma"])
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(
        concatenate([school_mu, school_sigma, shared_sigma, data_conditions]), repetitions
    )
    assert keras.ops.all(conditions == expected_output)

    conditions = prepare_inference_conditions(approximator, data, 2)

    # output has the correct shape
    expected_shape = (2, data.meta["N_classrooms"], data.meta["N_students"], 18)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    data_conditions = summary_outputs[0]
    school_mu = approximator.standardize_layers["school_mu"](data["school_mu"])
    school_sigma = approximator.standardize_layers["school_sigma"](data["school_sigma"])
    classroom_mu = approximator.standardize_layers["classroom_mu"](data["classroom_mu"])
    classroom_sigma = approximator.standardize_layers["classroom_sigma"](data["classroom_sigma"])
    shared_sigma = approximator.standardize_layers["shared_sigma"](data["shared_sigma"])
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(
        concatenate([school_mu, school_sigma, classroom_mu, classroom_sigma, shared_sigma, data_conditions]),
        repetitions,
    )
    assert keras.ops.all(conditions == expected_output)


def test_prepare_inference_conditions_crossed_design_irt(crossed_design_irt_simulator, crossed_design_irt_approximator):
    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    conditions = prepare_inference_conditions(approximator, data, 0)

    # output has the correct shape
    expected_shape = (2, 22)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[1]
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(data_conditions, repetitions)
    assert keras.ops.all(conditions == expected_output)

    conditions = prepare_inference_conditions(approximator, data, 1)

    # output has the correct shape
    expected_shape = (2, 26)
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[1]
    mu_question_mean = approximator.standardize_layers["mu_question_mean"](data["mu_question_mean"])
    sigma_question_mean = approximator.standardize_layers["sigma_question_mean"](data["sigma_question_mean"])
    mu_question_std = approximator.standardize_layers["mu_question_std"](data["mu_question_std"])
    sigma_question_std = approximator.standardize_layers["sigma_question_std"](data["sigma_question_std"])
    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(
        concatenate([mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std, data_conditions]),
        repetitions,
    )
    assert keras.ops.all(conditions == expected_output)

    conditions = prepare_inference_conditions(approximator, data, 2)

    # output has the correct shape
    expected_shape = (
        2,
        data.meta["num_students"],
        10 + data.meta["num_questions"] * 3 + 4 + 2,
    )
    assert keras.ops.shape(conditions) == expected_shape

    # output is as expected
    summary_outputs = summary_outputs_by_network(approximator, data)
    data_conditions = summary_outputs[0]
    mu_question_mean = approximator.standardize_layers["mu_question_mean"](data["mu_question_mean"])
    sigma_question_mean = approximator.standardize_layers["sigma_question_mean"](data["sigma_question_mean"])
    mu_question_std = approximator.standardize_layers["mu_question_std"](data["mu_question_std"])
    sigma_question_std = approximator.standardize_layers["sigma_question_std"](data["sigma_question_std"])

    question_mean = approximator.standardize_layers["question_mean"](data["question_mean"])
    question_mean_rank = keras.ops.ndim(question_mean)
    question_mean_perm = (*range(question_mean_rank - 2), question_mean_rank - 1, question_mean_rank - 2)
    question_mean_transpose = keras.ops.transpose(question_mean, axes=question_mean_perm)
    question_mean_reshape = keras.ops.reshape(
        question_mean_transpose, (*keras.ops.shape(question_mean_transpose)[:-2], -1)
    )

    question_std = approximator.standardize_layers["question_std"](data["question_std"])
    question_std_rank = keras.ops.ndim(question_std)
    question_std_perm = (*range(question_std_rank - 2), question_std_rank - 1, question_std_rank - 2)
    question_std_transpose = keras.ops.transpose(question_std, axes=question_std_perm)
    question_std_reshape = keras.ops.reshape(
        question_std_transpose, (*keras.ops.shape(question_std_transpose)[:-2], -1)
    )

    question_difficulty = approximator.standardize_layers["question_difficulty"](data["question_difficulty"])
    question_difficulty_rank = keras.ops.ndim(question_difficulty)
    question_difficulty_perm = (
        *range(question_difficulty_rank - 2),
        question_difficulty_rank - 1,
        question_difficulty_rank - 2,
    )
    question_difficulty_transpose = keras.ops.transpose(question_difficulty, axes=question_difficulty_perm)
    question_difficulty_reshape = keras.ops.reshape(
        question_difficulty_transpose, (*keras.ops.shape(question_mean_transpose)[:-2], -1)
    )

    repetitions = repetitions_from_data_shape(approximator, data_shapes)
    expected_output = add_node_reps_to_conditions(
        concatenate(
            [
                mu_question_mean,
                sigma_question_mean,
                mu_question_std,
                sigma_question_std,
                question_mean_reshape,
                question_std_reshape,
                question_difficulty_reshape,
                data_conditions,
            ]
        ),
        repetitions,
    )
    assert keras.ops.all(conditions == expected_output)
