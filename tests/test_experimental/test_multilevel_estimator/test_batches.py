import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.experimental.multilevel_estimator.batches import (
    MultilevelDataset,
    add_sequence_summaries,
    bucket,
    groupings,
    steps,
    to_batch,
)


def test_groupings(two_level_simulator, three_level_simulator, crossed_design_irt_simulator):
    assert groupings(two_level_simulator.graph, "y") == [(), ("locals",)]
    assert groupings(three_level_simulator.graph, "scores") == [(), ("classrooms",), ("classrooms", "students")]
    assert sorted(groupings(crossed_design_irt_simulator.graph, "observations")) == [(), ("questions",), ("students",)]


def test_groups(crossed_design_irt_simulator):
    graph = crossed_design_irt_simulator.graph.expand().invert()
    batch = to_batch(crossed_design_irt_simulator.sample(2), graph)
    groups = batch["observations"]["groups"]
    rows = batch["observations"]["indices"]

    assert sorted(groups) == ["dataset", "questions", "students"]
    np.testing.assert_array_equal(groups["dataset"]["ids"], rows["dataset"])
    np.testing.assert_array_equal(groups["dataset"]["counts"], np.bincount(rows["dataset"]))

    # every student answers every question, so a question has one answer per student for each school
    num_students = np.bincount(batch["students"]["indices"]["dataset"])
    answers_per_question = groups["questions"]["counts"][groups["questions"]["ids"]]
    np.testing.assert_array_equal(answers_per_question, num_students[rows["dataset"]])


def test_steps():
    # five questions, the first three in dataset 0 and the last two in dataset 1
    indices = {"dataset": np.array([0, 0, 0, 1, 1]), "questions": np.arange(5)}

    result = steps("questions", indices)

    # every dataset is one sequence, and each row knows its position in it
    assert result["ids"].tolist() == [0, 0, 0, 1, 1]
    assert result["counts"].tolist() == [3, 2]
    assert result["positions"].tolist() == [0, 1, 2, 0, 1]

    # every (earlier, later) pair of rows in the same sequence
    pairs = list(zip(result["earlier"].tolist(), result["later"].tolist()))
    assert pairs == [(0, 1), (0, 2), (1, 2), (3, 4)]


def test_only_unpadded_batches_have_sequence_steps(survey_simulator):
    graph = survey_simulator.graph.expand().invert()
    samples = survey_simulator.sample(2)

    unpadded = to_batch(samples, graph)
    padded = to_batch(samples, graph, padded=True)
    sequential = [node for node, entries in unpadded.items() if "steps" in entries]

    assert sequential
    for node in sequential:
        counts = unpadded[node]["steps"]["counts"]
        assert unpadded[node]["steps"]["sequence_steps"].tolist() == [list(range(max(counts)))]
        assert "sequence_steps" not in padded[node]["steps"]


def test_sequence_summaries():
    # five questions, the first three in dataset 0 and the last two in dataset 1
    indices = {"dataset": np.array([0, 0, 0, 1, 1]), "questions": np.arange(5)}
    values = np.array([[1.0], [2.0], [4.0], [10.0], [20.0]])
    batch = {"questions": {"values": {"difficulty": values}, "steps": steps("questions", indices)}}
    conditions = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])

    result = keras.ops.convert_to_numpy(add_sequence_summaries(batch, "questions", conditions))

    # the conditions, their mean over the sequence, the mean of the earlier values and their number
    expected = np.array(
        [
            [1.0, 2.0, 0.0, np.log(1)],
            [2.0, 2.0, 1.0, np.log(2)],
            [3.0, 2.0, 1.5, np.log(3)],
            [4.0, 4.5, 0.0, np.log(1)],
            [5.0, 4.5, 10.0, np.log(2)],
        ]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-6)


def test_bucket():
    sizes = set()

    for n in range(1, 1025):
        assert bucket(n) > n
        sizes.add(bucket(n))

    assert len(sizes) <= 4 * 11


def test_pad(two_level_simulator):
    simulation = two_level_simulator.sample(3)
    graph = two_level_simulator.graph.expand().invert()
    batch = to_batch(simulation, graph)
    padded = to_batch(simulation, graph, padded=True)

    for node, entries in batch.items():
        rows = len(entries["indices"][node])
        size = bucket(rows)

        for column, ids in entries["indices"].items():
            assert len(padded[node]["indices"][column]) == size
            np.testing.assert_array_equal(padded[node]["indices"][column][:rows], ids)

        for variable, values in entries["values"].items():
            padded_values = keras.ops.convert_to_numpy(padded[node]["values"][variable])
            assert len(padded_values) == size
            np.testing.assert_array_equal(padded_values[:rows], keras.ops.convert_to_numpy(values))

        weights = padded[node]["weights"]
        assert np.all(weights[rows:] == 0)
        np.testing.assert_allclose(np.mean(weights), 1.0, rtol=1e-6)


def test_dataset_yields_every_dataset_once_per_epoch(two_level_simulator):
    simulation = two_level_simulator.sample(5)
    graph = two_level_simulator.graph.expand().invert()
    dataset = MultilevelDataset(simulation, graph, batch_size=2, padded=False)

    assert len(dataset) == 3

    for _ in range(2):
        seen = []
        for i in range(len(dataset)):
            seen.extend(dataset[i]["hypers"]["values"]["hyper_mean"].ravel())
        dataset.on_epoch_end()

        np.testing.assert_array_equal(np.sort(seen), np.sort(simulation["hypers"].values["hyper_mean"].ravel()))


def test_dataset_applies_adapter_and_padding(two_level_simulator):
    simulation = two_level_simulator.sample(3)
    graph = two_level_simulator.graph.expand().invert()
    adapter = Adapter().log("hyper_std")

    padded = MultilevelDataset(simulation, graph, batch_size=3, adapter=adapter, padded=True)[0]
    unpadded = MultilevelDataset(simulation, graph, batch_size=3, adapter=adapter, padded=False)[0]
    raw = MultilevelDataset(simulation, graph, batch_size=3, padded=False)[0]

    assert "weights" in padded["hypers"]
    assert "weights" not in unpadded["hypers"]
    assert len(unpadded["hypers"]["indices"]["dataset"]) == 3

    expected = np.sort(np.log(raw["hypers"]["values"]["hyper_std"]).ravel())
    adapted = np.sort(keras.ops.convert_to_numpy(unpadded["hypers"]["values"]["hyper_std"]).ravel())
    np.testing.assert_allclose(adapted, expected, rtol=1e-5)


def test_dataset_pads_only_when_batch_sizes_can_change(two_level_simulator):
    graph = two_level_simulator.graph.expand().invert()
    simulation = two_level_simulator.sample(4)

    assert not MultilevelDataset(simulation, graph, batch_size=4).padded
    assert MultilevelDataset(simulation, graph, batch_size=2).padded
    assert MultilevelDataset(two_level_simulator, graph, batch_size=4, num_batches=1).padded
    assert "weights" not in MultilevelDataset(simulation, graph, batch_size=4)[0]["hypers"]
