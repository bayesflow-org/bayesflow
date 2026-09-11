import keras
import sympy as sp

from bayesflow.experimental.graphical_approximator.tensor_concatenation import concatenate


def test_resolve_shapes():
    from bayesflow.experimental.graphical_approximator.inference_variables import resolve_shapes

    a = (sp.Symbol("B"), sp.Symbol("N_groups"), 3)
    b = (sp.Symbol("B"), 3)

    assert resolve_shapes({"beta": a, "sigma": b}, {"B": 2, "N_groups": 3}) == {
        "beta": (2, 3, 3),
        "sigma": (2, 3),
    }
    assert resolve_shapes({"beta": a, "sigma": b}, {"N_groups": 3}) == {
        "beta": (sp.Symbol("B"), 3, 3),
        "sigma": (sp.Symbol("B"), 3),
    }


def test_inference_variables_by_network_missing_key(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variables_by_network

    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)
    single_level_approximator.build(data_shapes)

    result = inference_variables_by_network(single_level_approximator, {})
    assert result == {}


def test_inference_variables_by_network_single_level(single_level_simulator, single_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import (
        inference_variable_shapes_by_network,
        inference_variables_by_network,
    )

    data = single_level_simulator.sample(2)
    data_shapes = single_level_approximator._data_shapes(data)

    approximator = single_level_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    beta = approximator.standardize_layers["beta"](data["beta"], stage="validation")
    sigma = approximator.standardize_layers["sigma"](data["sigma"], stage="validation")

    expected_output = concatenate([beta, sigma])
    assert keras.ops.all(variables[0] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variables_by_network_two_level(two_level_simulator, two_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import (
        inference_variable_shapes_by_network,
        inference_variables_by_network,
    )

    data = two_level_simulator.sample(2)
    data_shapes = two_level_approximator._data_shapes(data)

    approximator = two_level_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    hyper_mean = approximator.standardize_layers["hyper_mean"](data["hyper_mean"], stage="validation")
    hyper_std = approximator.standardize_layers["hyper_std"](data["hyper_std"], stage="validation")
    shared_std = approximator.standardize_layers["shared_std"](data["shared_std"], stage="validation")

    expected_output = concatenate([hyper_mean, hyper_std, shared_std])
    assert keras.ops.all(variables[0] == expected_output)

    expected_output = approximator.standardize_layers["local_mean"](data["local_mean"], stage="validation")
    assert keras.ops.all(variables[1] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variables_by_network_three_level(three_level_simulator, three_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import (
        inference_variable_shapes_by_network,
        inference_variables_by_network,
    )

    data = three_level_simulator.sample(2)
    data_shapes = three_level_approximator._data_shapes(data)

    approximator = three_level_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    school_mu = approximator.standardize_layers["school_mu"](data["school_mu"], stage="validation")
    school_sigma = approximator.standardize_layers["school_sigma"](data["school_sigma"], stage="validation")
    shared_sigma = approximator.standardize_layers["shared_sigma"](data["shared_sigma"], stage="validation")

    expected_output = concatenate([school_mu, school_sigma, shared_sigma])
    assert keras.ops.all(variables[0] == expected_output)

    classroom_mu = approximator.standardize_layers["classroom_mu"](data["classroom_mu"], stage="validation")
    classroom_sigma = approximator.standardize_layers["classroom_sigma"](data["classroom_sigma"], stage="validation")

    expected_output = concatenate([classroom_mu, classroom_sigma])
    assert keras.ops.all(variables[1] == expected_output)

    student_mu = approximator.standardize_layers["student_mu"](data["student_mu"], stage="validation")
    student_sigma = approximator.standardize_layers["student_sigma"](data["student_sigma"], stage="validation")

    expected_output = concatenate([student_mu, student_sigma])
    assert keras.ops.all(variables[2] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variables_by_network_crossed_design_irt(
    crossed_design_irt_simulator, crossed_design_irt_approximator
):
    from bayesflow.experimental.graphical_approximator.inference_variables import (
        inference_variable_shapes_by_network,
        inference_variables_by_network,
    )

    data = crossed_design_irt_simulator.sample(2)
    data_shapes = crossed_design_irt_approximator._data_shapes(data)

    approximator = crossed_design_irt_approximator
    approximator.build(data_shapes)

    variables = inference_variables_by_network(approximator, data)

    mu_question_mean = approximator.standardize_layers["mu_question_mean"](data["mu_question_mean"], stage="validation")
    sigma_question_mean = approximator.standardize_layers["sigma_question_mean"](
        data["sigma_question_mean"], stage="validation"
    )
    mu_question_std = approximator.standardize_layers["mu_question_std"](data["mu_question_std"], stage="validation")
    sigma_question_std = approximator.standardize_layers["sigma_question_std"](
        data["sigma_question_std"], stage="validation"
    )

    expected_output = concatenate([mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std])
    assert keras.ops.all(variables[0] == expected_output)

    question_mean = approximator.standardize_layers["question_mean"](data["question_mean"], stage="validation")
    question_std = approximator.standardize_layers["question_std"](data["question_std"], stage="validation")
    question_difficulty = approximator.standardize_layers["question_difficulty"](
        data["question_difficulty"], stage="validation"
    )

    expected_output = concatenate([question_mean, question_std, question_difficulty])
    assert keras.ops.all(variables[1] == expected_output)

    expected_output = approximator.standardize_layers["student_ability"](data["student_ability"], stage="validation")
    assert keras.ops.all(variables[2] == expected_output)

    variable_shapes = inference_variable_shapes_by_network(approximator, data_shapes)

    for k, v in variables.items():
        assert variable_shapes[k] == v.shape


def test_inference_variable_shapes_by_network_single_level(single_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variable_shapes_by_network

    expected_shapes = {
        0: (sp.Symbol("B"), 3),  # 3 variables (beta (2-dimensional), sigma)
    }
    assert inference_variable_shapes_by_network(single_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_two_level(two_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variable_shapes_by_network

    expected_shapes = {
        0: (sp.Symbol("B"), 3),  # 3 variables (hyper_mean, hyper_std, shared_std)
        1: (sp.Symbol("B"), 6, 1),  # 1 variable (local_mean)
    }
    assert inference_variable_shapes_by_network(two_level_approximator) == expected_shapes


def test_inference_variable_shapes_by_network_three_level(three_level_approximator):
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variable_shapes_by_network

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
    from bayesflow.experimental.graphical_approximator.inference_variables import inference_variable_shapes_by_network

    expected_shapes = {
        # 4 variables (mu_question_mean, sigma_question_mean, mu_question_std, sigma_question_std)
        0: (sp.Symbol("B"), 4),
        # 3 variables (question_mean, question_std, question_difficulty)
        1: (sp.Symbol("B"), sp.Symbol("num_questions"), 3),
        # 1 variable (student_ability)
        2: (sp.Symbol("B"), sp.Symbol("num_students"), 1),
    }
    assert inference_variable_shapes_by_network(crossed_design_irt_approximator) == expected_shapes
