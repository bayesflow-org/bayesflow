import keras
import numpy as np
import pytest

from bayesflow.experimental.multilevel_estimator import MultilevelEstimator
from bayesflow.experimental.multilevel_estimator.encoder import MultilevelEncoder
from bayesflow.networks import CouplingFlow


def small_estimator(simulator):
    # encoders of width 4, so that the widths of the conditions are easy to count
    graph = simulator.graph.expand().invert()
    simulation_graph = graph.expanded_graph.simulation_graph

    inference_networks = []
    encoders = []
    for target_nodes in graph.networks():
        network_encoders = {}
        for node in graph.conditions(target_nodes):
            if simulation_graph.out_degree(node) == 0:
                network_encoders[node] = MultilevelEncoder(width=4)

        inference_networks.append(CouplingFlow())
        encoders.append(network_encoders)

    return MultilevelEstimator(graph, inference_networks=inference_networks, encoders=encoders)


def all_conditions(estimator, batch):
    conditions = []

    for i in range(len(estimator.networks)):
        conditions.append(keras.ops.convert_to_numpy(estimator.network_conditions(batch, i)))

    return conditions


@pytest.mark.parametrize(
    "simulator",
    [
        "single_level_simulator",
        "two_level_simulator",
        "three_level_simulator",
        "crossed_design_irt_simulator",
        "two_data_nodes_simulator",
    ],
)
def test_one_condition_row_per_inferred_row(simulator, request):
    simulator = request.getfixturevalue(simulator)
    estimator = small_estimator(simulator)
    batch = estimator.to_batch(simulator.sample(3))

    conditions = all_conditions(estimator, batch)

    for i, target_nodes in enumerate(estimator.networks):
        target = target_nodes[0]
        assert len(conditions[i]) == len(batch[target]["indices"][target])


def test_condition_widths(two_level_simulator, two_data_nodes_simulator):
    estimator = small_estimator(two_level_simulator)
    conditions = all_conditions(estimator, estimator.to_batch(two_level_simulator.sample(2)))

    # hypers and shared: encoded y (4) and its number of rows (1)
    assert conditions[0].shape == (2, 5)
    # locals: encoded y (4) and its number of rows (1), hyper_mean and hyper_std (2)
    assert conditions[1].shape == (12, 7)

    estimator = small_estimator(two_data_nodes_simulator)
    conditions = all_conditions(estimator, estimator.to_batch(two_data_nodes_simulator.sample(2)))

    # every data node is encoded separately
    assert conditions[0].shape == (2, 10)
    assert conditions[1].shape == (12, 12)


def test_conditions_depend_on_which_observations_share_a_group(two_level_simulator):
    estimator = small_estimator(two_level_simulator)
    batch = estimator.to_batch(two_level_simulator.sample(2))
    before = all_conditions(estimator, batch)

    # swap an observation of the first group with one of the second group, both in dataset 0,
    # so the dataset has the same observations, but they belong to different groups
    groups = batch["y"]["indices"]["locals"]
    first = np.flatnonzero(groups == 0)[0]
    second = np.flatnonzero(groups == 1)[0]

    y = keras.ops.convert_to_numpy(batch["y"]["values"]["y"])
    y[[first, second]] = y[[second, first]]
    batch["y"]["values"]["y"] = y
    after = all_conditions(estimator, batch)

    # the conditions of hypers and shared have one row per dataset: only dataset 0 changes
    hypers = estimator.networks.index(["hypers", "shared"])
    assert not np.allclose(before[hypers][0], after[hypers][0])
    np.testing.assert_allclose(before[hypers][1], after[hypers][1], rtol=1e-4, atol=1e-4)


def test_conditions_are_paired_with_the_data(crossed_design_irt_simulator):
    estimator = small_estimator(crossed_design_irt_simulator)
    batch = estimator.to_batch(crossed_design_irt_simulator.sample(2))
    before = all_conditions(estimator, batch)

    # swap the parameters of the first two questions of dataset 0. Every student answered both,
    # so each student still sees the same question parameters, but the answers belong to the other question
    first, second = np.flatnonzero(batch["questions"]["indices"]["dataset"] == 0)[:2]

    for variable, values in batch["questions"]["values"].items():
        values = keras.ops.convert_to_numpy(values)
        values[[first, second]] = values[[second, first]]
        batch["questions"]["values"][variable] = values

    after = all_conditions(estimator, batch)

    # the conditions of students have one row per student: only the students of dataset 0 change
    students = estimator.networks.index(["students"])
    in_dataset_0 = batch["students"]["indices"]["dataset"] == 0
    assert not np.allclose(before[students][in_dataset_0], after[students][in_dataset_0])
    np.testing.assert_allclose(before[students][~in_dataset_0], after[students][~in_dataset_0], rtol=1e-4, atol=1e-4)
