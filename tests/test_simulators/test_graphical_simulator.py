import numpy as np

import bayesflow as bf


def test_single_level_simulator(single_level_simulator):
    # prior -> likelihood

    simulator = single_level_simulator
    assert isinstance(simulator, bf.experimental.graphical_simulator.GraphicalSimulator)
    assert isinstance(simulator.sample(5), dict)

    samples = simulator.sample(12)
    expected_keys = ["N", "beta", "sigma", "x", "y"]

    assert set(samples.keys()) == set(expected_keys)
    assert 5 <= samples["N"] < 15

    # prior node
    assert np.shape(samples["beta"]) == (12, 2)  # num_samples, beta_dim
    assert np.shape(samples["sigma"]) == (12, 1)  # num_samples, sigma_dim

    # likelihood node
    assert np.shape(samples["x"]) == (12, samples["N"])
    assert np.shape(samples["y"]) == (12, samples["N"])


def test_two_level_simulator(two_level_simulator):
    # hypers
    #   |
    # locals  shared
    #     \    /
    #      \  /
    #       y

    simulator = two_level_simulator
    assert isinstance(simulator, bf.experimental.graphical_simulator.GraphicalSimulator)
    assert isinstance(simulator.sample(5), dict)

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


def test_two_level_repeated_roots_simulator(two_level_repeated_roots_simulator):
    # hypers
    #   |
    # locals  shared
    #     \    /
    #      \  /
    #       y

    simulator = two_level_repeated_roots_simulator
    assert isinstance(simulator, bf.experimental.graphical_simulator.GraphicalSimulator)
    assert isinstance(simulator.sample(5), dict)

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


def test_crossed_design_irt_simulator(crossed_design_irt_simulator):
    #  schools
    #   /     \
    # exams  students
    #   |       |
    # questions |
    #    \     /
    #  observations

    simulator = crossed_design_irt_simulator
    assert isinstance(simulator, bf.experimental.graphical_simulator.GraphicalSimulator)
    assert isinstance(simulator.sample(5), dict)

    samples = simulator.sample(22)
    expected_keys = [
        "mu_exam_mean",
        "sigma_exam_mean",
        "mu_exam_std",
        "sigma_exam_std",
        "exam_mean",
        "exam_std",
        "question_difficulty",
        "student_ability",
        "obs",
        "num_exams",  # np.random.randint(2, 4)
        "num_questions",  # np.random.randint(10, 21)
        "num_students",  # np.random.randint(100, 201)
    ]

    assert set(samples.keys()) == set(expected_keys)

    # schools node
    assert np.shape(samples["mu_exam_mean"]) == (22, 1)
    assert np.shape(samples["sigma_exam_mean"]) == (22, 1)
    assert np.shape(samples["mu_exam_std"]) == (22, 1)
    assert np.shape(samples["sigma_exam_std"]) == (22, 1)

    # exams node
    assert np.shape(samples["exam_mean"]) == (22, samples["num_exams"], 1)
    assert np.shape(samples["exam_std"]) == (22, samples["num_exams"], 1)

    # questions node
    assert np.shape(samples["question_difficulty"]) == (22, samples["num_exams"], samples["num_questions"], 1)

    # students node
    assert np.shape(samples["student_ability"]) == (22, samples["num_students"], 1)

    # observations node
    assert np.shape(samples["obs"]) == (
        22,
        samples["num_exams"],
        samples["num_students"],
        samples["num_questions"],
        1,
    )
