import numpy as np
import pytest


def estimator_for(simulator, network=None):
    from bayesflow.experimental.multilevel_estimator import MultilevelEstimator
    from bayesflow.networks import CouplingFlow

    network = network or CouplingFlow
    graph = simulator.graph.expand().invert()

    inference_networks = []
    for target_nodes in graph.networks():
        inference_networks.append(network())

    return MultilevelEstimator(graph, inference_networks=inference_networks)


# a survey, where every square is surveyed in one window of years, so obs has no row for most (square, year) pairs
def squares_reps(size):
    return np.random.randint(1, 5, size=size)


def surveyed(combinations, num_years):
    _, first_row, square = np.unique(combinations["squares"], return_index=True, return_inverse=True)
    year = combinations["years"] % num_years
    years_of_square = num_years[first_row]

    first = np.random.randint(0, years_of_square)
    last = first + np.random.randint(0, years_of_square - first)

    return ((year >= first[square]) & (year <= last[square])).astype(int)


@pytest.fixture()
def survey_simulator():
    from bayesflow.experimental.multilevel_simulator import MultilevelSimulator

    simulator = MultilevelSimulator(meta_fn=lambda: {"num_years": 5})
    simulator.add_node("globals", sample_fn=lambda: {"mu": np.random.normal()})
    simulator.add_node("squares", sample_fn=lambda mu: {"alpha": np.random.normal(mu)}, reps=squares_reps)
    simulator.add_node("years", sample_fn=lambda: {"beta": np.random.normal()}, reps="num_years")
    simulator.add_node("obs", sample_fn=lambda alpha, beta: {"count": np.random.normal(alpha + beta)}, reps=surveyed)
    simulator.add_edge("globals", "squares")
    simulator.add_edge("globals", "years")
    simulator.add_edge("squares", "obs")
    simulator.add_edge("years", "obs")

    return simulator


@pytest.fixture()
def survey_estimator(survey_simulator):
    return estimator_for(survey_simulator)


@pytest.fixture()
def single_level_simulator():
    from bayesflow.experimental.multilevel_simulator.example_simulators import single_level_simulator

    return single_level_simulator()


@pytest.fixture()
def single_level_estimator(single_level_simulator):
    return estimator_for(single_level_simulator)


@pytest.fixture()
def two_level_simulator():
    from bayesflow.experimental.multilevel_simulator.example_simulators import two_level_simulator

    return two_level_simulator()


@pytest.fixture()
def two_level_estimator(two_level_simulator):
    return estimator_for(two_level_simulator)


# reps are saved with the estimator, so they must be a top-level function
def w_reps(size):
    return np.random.randint(3, 6, size=size)


@pytest.fixture()
def two_data_nodes_simulator():
    from bayesflow.experimental.multilevel_simulator.example_simulators import two_level_simulator

    # a second data node below locals
    simulator = two_level_simulator()
    simulator.add_node("w", sample_fn=lambda local_mean: {"w": np.random.normal(local_mean, 0.5, size=2)}, reps=w_reps)
    simulator.add_edge("locals", "w")

    return simulator


@pytest.fixture()
def two_data_nodes_estimator(two_data_nodes_simulator):
    return estimator_for(two_data_nodes_simulator)


@pytest.fixture()
def three_level_simulator():
    from bayesflow.experimental.multilevel_simulator.example_simulators import three_level_simulator

    return three_level_simulator()


@pytest.fixture()
def three_level_estimator(three_level_simulator):
    return estimator_for(three_level_simulator)


@pytest.fixture()
def crossed_design_irt_simulator():
    from bayesflow.experimental.multilevel_simulator.example_simulators import crossed_design_irt_simulator

    return crossed_design_irt_simulator()


@pytest.fixture()
def crossed_design_irt_estimator(crossed_design_irt_simulator):
    return estimator_for(crossed_design_irt_simulator)
