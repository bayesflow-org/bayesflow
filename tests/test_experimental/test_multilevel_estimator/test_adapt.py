import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.experimental.multilevel_estimator import MultilevelEstimator
from bayesflow.experimental.multilevel_estimator.batches import to_batch
from bayesflow.networks import CouplingFlow


@pytest.fixture()
def adapter():
    return Adapter().log(["hyper_std", "shared_std"]).constrain("local_mean", lower=-10.0, upper=10.0)


@pytest.fixture()
def estimator(adapter, two_level_simulator):
    graph = two_level_simulator.graph.expand().invert()
    inference_networks = [CouplingFlow() for _ in graph.networks()]

    return MultilevelEstimator(graph, inference_networks=inference_networks, adapter=adapter)


@pytest.fixture()
def batch(estimator, two_level_simulator):
    return to_batch(two_level_simulator.sample(3), estimator.graph)


def per_dataset(values, datasets):
    return np.bincount(datasets, weights=values, minlength=3)


def test_values_match_adapter_per_node(batch, adapter, estimator):
    adapted = estimator.adapt(batch)

    for node, entries in batch.items():
        expected = adapter(entries["values"], strict=False)

        for variable, value in expected.items():
            np.testing.assert_allclose(keras.ops.convert_to_numpy(adapted[node]["values"][variable]), value)


def test_indices_are_unchanged(batch, estimator):
    adapted = estimator.adapt(batch)

    for node, entries in batch.items():
        for column, ids in entries["indices"].items():
            assert adapted[node]["indices"][column] is ids

        assert adapted[node].get("groups") is entries.get("groups")


def test_inverse_restores_values(batch, estimator):
    restored = estimator.adapt(estimator.adapt(batch), inverse=True)

    for node, entries in batch.items():
        for variable, value in entries["values"].items():
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(restored[node]["values"][variable]), value, rtol=1e-4, atol=1e-5
            )


def test_jacobian_is_summed_per_dataset(batch, adapter, estimator):
    _, jacobians = estimator.adapt(batch, log_det_jac=True)

    for node, entries in batch.items():
        _, node_jacobians = adapter(entries["values"], strict=False, log_det_jac=True)
        expected = np.zeros(3)

        for value in node_jacobians.values():
            expected += per_dataset(keras.ops.convert_to_numpy(value), entries["indices"]["dataset"])

        np.testing.assert_allclose(keras.ops.convert_to_numpy(jacobians[node]), expected, rtol=1e-5)


def test_works_on_tensors(batch, estimator):
    tensors = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    adapted, jacobians = estimator.adapt(tensors, log_det_jac=True)

    for node, entries in adapted.items():
        assert keras.ops.shape(jacobians[node]) == (3,)

        for ids in entries["indices"].values():
            assert "int" in keras.backend.standardize_dtype(ids.dtype)


def test_to_batch_with_adapter_matches_adapt(estimator, adapter, two_level_simulator):
    simulation = two_level_simulator.sample(3)
    expected = estimator.adapt(to_batch(simulation, estimator.graph))
    adapted = to_batch(simulation, estimator.graph, adapter)

    for node, entries in expected.items():
        for variable, value in entries["values"].items():
            np.testing.assert_allclose(
                keras.ops.convert_to_numpy(adapted[node]["values"][variable]), keras.ops.convert_to_numpy(value)
            )
