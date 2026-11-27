import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.experimental.multilevel_estimator import MultilevelEstimator
from bayesflow.experimental.multilevel_estimator.batches import tile_batch, to_batch
from bayesflow.networks import CouplingFlow


@pytest.fixture(params=["two_level_simulator", "survey_simulator"])
def simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def estimator(simulator):
    graph = simulator.graph.expand().invert()
    adapter = Adapter().log(["hyper_std", "shared_std"]) if "hypers" in simulator.graph else Adapter()
    estimator = MultilevelEstimator(
        graph, inference_networks=[CouplingFlow() for _ in graph.networks()], adapter=adapter
    )
    estimator.build_from_data(estimator.to_batch(simulator.sample(2)))

    return estimator


def latent_nodes(estimator):
    return [node for target_nodes in estimator.networks for node in target_nodes]


def test_log_prob_of_batch_matches_graph_samples(estimator, simulator):
    data = simulator.sample(3)
    batch = to_batch(data, estimator.graph)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(estimator.log_prob(batch)), estimator.log_prob(data), rtol=1e-5, atol=1e-4
    )


def test_log_prob_of_copies_matches_original(estimator, simulator):
    batch = to_batch(simulator.sample(2), estimator.graph)
    copies = 3

    log_prob = keras.ops.convert_to_numpy(estimator.log_prob(batch))
    tiled = keras.ops.convert_to_numpy(estimator.log_prob(tile_batch(batch, copies)))

    np.testing.assert_allclose(tiled, np.repeat(log_prob, copies), rtol=1e-5, atol=1e-4)


def test_sample_from_batch(estimator, simulator):
    batch = to_batch(simulator.sample(2), estimator.graph)
    num_samples = 4

    samples = estimator.sample(num_samples=num_samples, conditions=batch)

    assert sorted(samples) == sorted(latent_nodes(estimator))

    for node, values in samples.items():
        rows = len(batch[node]["indices"][node])

        for variable, value in values.items():
            value = keras.ops.convert_to_numpy(value)
            assert value.shape[0] == num_samples * rows
            assert np.all(np.isfinite(value))

            if variable in ("hyper_std", "shared_std"):
                assert np.all(value > 0)


def test_sample_rejects_padded_batch(estimator, simulator):
    batch = to_batch(simulator.sample(2), estimator.graph, padded=True)

    with pytest.raises(ValueError, match="padded"):
        estimator.sample(num_samples=2, conditions=batch)


def test_log_prob_of_joined_samples(estimator, simulator):
    batch = to_batch(simulator.sample(2), estimator.graph)
    num_samples = 3

    samples = estimator.sample(num_samples=num_samples, conditions=batch)
    joined = estimator.join_samples(batch, samples, num_samples)
    log_prob = keras.ops.convert_to_numpy(estimator.log_prob(joined))

    assert log_prob.shape == (2 * num_samples,)
    assert np.all(np.isfinite(log_prob))
