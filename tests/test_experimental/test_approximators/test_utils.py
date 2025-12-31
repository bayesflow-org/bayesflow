from bayesflow.experimental.graphical_approximator.utils import (
    expand_shape_rank,
    stack_shapes,
    concatenate_shapes,
    add_sample_dimension,
    concatenate,
    repetitions_from_data_shape,
    inference_condition_shapes_by_network,
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
    # {0: (2, N_schools, N_students, 10), 1: (2, N_classrooms, 20), 2: (2, 30)}
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
        ),  # 10 summary dimensions + num_questions * 3 + 4 variables + 2 node reps
    }

    assert inference_condition_shapes_by_network(crossed_design_irt_approximator, data_shapes) == expected_shapes
