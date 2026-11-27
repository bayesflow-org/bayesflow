import keras
import numpy as np
import pytest

from bayesflow.experimental.multilevel_estimator.batches import tile_batch, to_batch
from bayesflow.experimental.multilevel_simulator.samples import concatenate


SIMULATORS = ["two_level_simulator", "crossed_design_irt_simulator", "survey_simulator"]


def as_numpy(batch):
    return keras.tree.map_structure(keras.ops.convert_to_numpy, batch)


def assert_same_batch(batch, expected):
    assert sorted(batch) == sorted(expected)

    for node in expected:
        assert sorted(batch[node]) == sorted(expected[node])

        for key, value in expected[node].items():
            flat = keras.tree.flatten(value)
            tiled = keras.tree.flatten(batch[node][key])
            assert len(flat) == len(tiled)

            for a, b in zip(tiled, flat):
                if np.issubdtype(np.asarray(b).dtype, np.integer):
                    np.testing.assert_array_equal(a, b)
                else:
                    np.testing.assert_allclose(a, b, rtol=1e-6)


@pytest.mark.parametrize("simulator_name", SIMULATORS)
def test_single_dataset_matches_concatenated_copies(simulator_name, request):
    simulator = request.getfixturevalue(simulator_name)
    graph = simulator.graph.expand().invert()
    observed = simulator.sample(1)

    tiled = as_numpy(tile_batch(to_batch(observed, graph), 4))
    expected = to_batch(concatenate([observed] * 4), graph)

    assert_same_batch(tiled, expected)


@pytest.mark.parametrize("simulator_name", SIMULATORS)
def test_copies_of_several_datasets(simulator_name, request):
    simulator = request.getfixturevalue(simulator_name)
    graph = simulator.graph.expand().invert()
    batch = to_batch(simulator.sample(2), graph)
    copies = 3

    tiled = as_numpy(tile_batch(batch, copies))

    for node, entries in batch.items():
        rows = len(entries["indices"][node])
        tiled_entries = tiled[node]
        copy = np.repeat(np.arange(copies), rows)

        np.testing.assert_array_equal(
            tiled_entries["indices"]["dataset"], np.tile(entries["indices"]["dataset"], copies) * copies + copy
        )

        for column, ids in entries["indices"].items():
            if column == "dataset":
                continue

            for variable, value in batch[column]["values"].items():
                original = value[ids]
                through_tiled = tiled[column]["values"][variable][tiled_entries["indices"][column]]
                np.testing.assert_allclose(through_tiled, np.concatenate([original] * copies), rtol=1e-6)

            parent_rows = len(batch[column]["indices"][column])
            assert np.all(tiled_entries["indices"][column] // parent_rows == copy)

        for name, group in entries.get("groups", {}).items():
            tiled_group = tiled_entries["groups"][name]
            np.testing.assert_array_equal(
                tiled_group["counts"][tiled_group["ids"]], np.tile(group["counts"][group["ids"]], copies)
            )

        if "steps" in entries:
            steps = entries["steps"]
            tiled_steps = tiled_entries["steps"]
            np.testing.assert_array_equal(tiled_steps["positions"], np.tile(steps["positions"], copies))
            np.testing.assert_array_equal(tiled_steps["sequence_steps"], steps["sequence_steps"])
            assert np.all(tiled_steps["earlier"] // rows == tiled_steps["later"] // rows)
            assert len(tiled_steps["earlier"]) == copies * len(steps["earlier"])


def test_works_on_tensors(two_level_simulator):
    graph = two_level_simulator.graph.expand().invert()
    batch = to_batch(two_level_simulator.sample(2), graph)
    tensors = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)

    tiled = tile_batch(tensors, 3)

    assert keras.ops.shape(tiled["hypers"]["indices"]["dataset"]) == (6,)
    assert "int" in keras.backend.standardize_dtype(tiled["y"]["indices"]["dataset"].dtype)
    assert_same_batch(as_numpy(tiled), as_numpy(tile_batch(batch, 3)))
