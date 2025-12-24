import numpy as np
import pytest

from bayesflow.experimental.graphical_simulator import GraphicalSimulator, SimulationOutput


def test_single_level_simulator(single_level_simulator):
    # prior -> likelihood

    simulator = single_level_simulator
    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(simulator.sample(5), SimulationOutput)

    samples = simulator.sample(12)
    expected_keys = ["beta", "sigma", "x", "y"]

    assert set(samples.keys()) == set(expected_keys)
    assert 5 <= samples.meta["N"] < 15

    # prior node
    assert np.shape(samples["beta"]) == (12, 2)  # num_samples, beta_dim
    assert np.shape(samples["sigma"]) == (12, 1)  # num_samples, sigma_dim

    # likelihood node
    assert np.shape(samples["x"]) == (12, samples.meta["N"], 1)
    assert np.shape(samples["y"]) == (12, samples.meta["N"], 1)

    # variable names
    assert simulator.variable_names() == {"prior": ["beta", "sigma"], "likelihood": ["x", "y"]}


def test_two_level_simulator(two_level_simulator):
    # hypers
    #   |
    # locals  shared
    #     \    /
    #      \  /
    #       y

    simulator = two_level_simulator
    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(simulator.sample(5), SimulationOutput)

    samples = simulator.sample(15)
    expected_keys = ["hyper_mean", "hyper_std", "local_mean", "shared_std", "y"]

    assert set(samples.keys()) == set(expected_keys)

    # hypers node
    assert np.shape(samples["hyper_mean"]) == (15, 1)
    assert np.shape(samples["hyper_std"]) == (15, 1)

    # locals node
    assert np.shape(samples["local_mean"]) == (15, 6, 1)

    # shared node
    assert np.shape(samples["shared_std"]) == (15, 1)

    # y node
    assert np.shape(samples["y"]) == (15, 6, 10, 1)

    # variable names
    assert simulator.variable_names() == {
        "hypers": ["hyper_mean", "hyper_std"],
        "locals": ["local_mean"],
        "shared": ["shared_std"],
        "y": ["y"],
    }


def test_two_level_repeated_roots_simulator(two_level_repeated_roots_simulator):
    # hypers
    #   |
    # locals  shared
    #     \    /
    #      \  /
    #       y

    simulator = two_level_repeated_roots_simulator
    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(simulator.sample(5), SimulationOutput)

    samples = simulator.sample(15)
    expected_keys = ["hyper_mean", "hyper_std", "local_mean", "shared_std", "y"]

    assert set(samples.keys()) == set(expected_keys)

    # hypers node
    assert np.shape(samples["hyper_mean"]) == (15, 5, 1)
    assert np.shape(samples["hyper_std"]) == (15, 5, 1)

    # locals node
    assert np.shape(samples["local_mean"]) == (15, 5, 6, 1)

    # shared node
    assert np.shape(samples["shared_std"]) == (15, 1)

    # y node
    assert np.shape(samples["y"]) == (15, 5, 6, 10, 1)

    # variable names
    assert simulator.variable_names() == {
        "hypers": ["hyper_mean", "hyper_std"],
        "locals": ["local_mean"],
        "shared": ["shared_std"],
        "y": ["y"],
    }


def test_three_level_simulator(three_level_simulator):
    #  schools
    #     |
    #     |
    # classrooms
    #     |
    #     |     shared
    # students    /
    #      \     /
    #       \   /
    #      scores

    simulator = three_level_simulator
    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(simulator.sample(5), SimulationOutput)

    samples = simulator.sample(15)
    expected_keys = [
        "school_mu",
        "school_sigma",
        "classroom_mu",
        "classroom_sigma",
        "student_mu",
        "student_sigma",
        "shared_sigma",
        "y",
    ]
    expected_meta_keys = ["N_classrooms", "N_students", "N_scores"]

    assert set(samples.keys()) == set(expected_keys)
    assert set(samples.meta.keys()) == set(expected_meta_keys)

    # schools node
    assert np.shape(samples["school_mu"]) == (15, 1)
    assert np.shape(samples["school_sigma"]) == (15, 1)

    # classrooms node
    assert np.shape(samples["classroom_mu"]) == (15, samples.meta["N_classrooms"], 1)
    assert np.shape(samples["classroom_sigma"]) == (15, samples.meta["N_classrooms"], 1)

    # students node
    assert np.shape(samples["student_mu"]) == (15, samples.meta["N_classrooms"], samples.meta["N_students"], 1)
    assert np.shape(samples["student_sigma"]) == (15, samples.meta["N_classrooms"], samples.meta["N_students"], 1)

    # shared node
    assert np.shape(samples["shared_sigma"]) == (15, 1)

    # y node
    assert np.shape(samples["y"]) == (
        15,
        samples.meta["N_classrooms"],
        samples.meta["N_students"],
        samples.meta["N_scores"],
        1,
    )

    # variable names
    assert simulator.variable_names() == {
        "schools": ["school_mu", "school_sigma"],
        "classrooms": ["classroom_mu", "classroom_sigma"],
        "students": ["student_mu", "student_sigma"],
        "shared": ["shared_sigma"],
        "scores": ["y"],
    }


def test_crossed_design_irt_simulator(crossed_design_irt_simulator):
    #  schools
    #   /     \
    #   |  students
    #   |       |
    # questions |
    #    \     /
    #  observations

    simulator = crossed_design_irt_simulator
    assert isinstance(simulator, GraphicalSimulator)
    assert isinstance(simulator.sample(5), SimulationOutput)

    samples = simulator.sample(22)
    expected_keys = [
        "mu_question_mean",
        "sigma_question_mean",
        "mu_question_std",
        "sigma_question_std",
        "question_mean",
        "question_std",
        "question_difficulty",
        "student_ability",
        "obs",
    ]
    expected_meta_keys = [
        "num_questions",  # 15
        "num_students",  # np.random.randint(100, 201)
    ]

    assert set(samples.keys()) == set(expected_keys)
    assert set(samples.meta.keys()) == set(expected_meta_keys)

    # schools node
    assert np.shape(samples["mu_question_mean"]) == (22, 1)
    assert np.shape(samples["sigma_question_mean"]) == (22, 1)
    assert np.shape(samples["mu_question_std"]) == (22, 1)
    assert np.shape(samples["sigma_question_std"]) == (22, 1)

    # questions node
    assert np.shape(samples["question_mean"]) == (22, samples.meta["num_questions"], 1)
    assert np.shape(samples["question_std"]) == (22, samples.meta["num_questions"], 1)
    assert np.shape(samples["question_difficulty"]) == (
        22,
        samples.meta["num_questions"],
        1,
    )

    # students node
    assert np.shape(samples["student_ability"]) == (22, samples.meta["num_students"], 1)

    # observations node
    assert np.shape(samples["obs"]) == (
        22,
        samples.meta["num_questions"],
        samples.meta["num_students"],
        1,
    )

    # variable names
    assert simulator.variable_names() == {
        "schools": ["mu_question_mean", "sigma_question_mean", "mu_question_std", "sigma_question_std"],
        "questions": ["question_mean", "question_std", "question_difficulty"],
        "students": ["student_ability"],
        "observations": ["obs"],
    }


def test_simulation_output():
    test_data = {"a": 1, "b": 2, "c": 3}
    meta_data = {"d": 4, "e": 5, "f": 6}

    output = SimulationOutput(test_data, meta_data)

    assert set(output.keys()) == set(test_data.keys())
    assert set(output.meta.keys()) == set(meta_data.keys())

    # copy method
    copied = output.copy()
    assert copied == output
    assert copied is not output

    # __getitem__
    assert output["a"] == 1
    with pytest.raises(KeyError):
        output["d"]

    # __setitem__
    output["z"] = 5
    assert output["z"] == 5

    # __delitem__
    del output["z"]
    with pytest.raises(KeyError):
        output["z"]

    # __iter__
    for k, v in output.items():
        assert k in output.keys()
        assert v in output.values()

    # __len__
    assert len(output) == 3
