import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.experimental.multilevel_estimator import MultilevelEstimator
from bayesflow.experimental.multilevel_estimator.encoder import MultilevelEncoder
from bayesflow.experimental.multilevel_simulator.samples import concatenate, to_tensors
from bayesflow.masking import MaskHandler
from bayesflow.networks import ConsistencyModel, CouplingFlow, DiffusionModel, FlowMatching, StableConsistencyModel

from .conftest import estimator_for


EXAMPLES = [
    ("single_level_simulator", "single_level_estimator"),
    ("two_level_simulator", "two_level_estimator"),
    ("three_level_simulator", "three_level_estimator"),
    ("two_data_nodes_simulator", "two_data_nodes_estimator"),
    ("crossed_design_irt_simulator", "crossed_design_irt_estimator"),
    ("survey_simulator", "survey_estimator"),
]


@pytest.mark.parametrize(("simulator", "estimator"), EXAMPLES)
def test_fit_sample(simulator, estimator, request):
    simulator = request.getfixturevalue(simulator)
    estimator = request.getfixturevalue(estimator)
    estimator.compile()

    metrics = estimator.compute_metrics(**estimator.to_batch(simulator.sample(2)))
    assert "loss" in metrics

    estimator.fit(simulator=simulator, batch_size=2, num_batches=1, epochs=1)
    estimator.fit(dataset=simulator.sample(4), batch_size=2, epochs=1)

    observed = simulator.sample(3)
    assert_samples_match(estimator, observed, estimator.sample(num_samples=5, conditions=observed), 5)

    log_prob = estimator.log_prob(observed)
    assert log_prob.shape == (3,)
    assert np.all(np.isfinite(log_prob))


def assert_samples_match(estimator, observed, samples, num_samples):
    latent_nodes = set()
    for target_nodes in estimator.networks:
        for node in target_nodes:
            latent_nodes.add(node)

            for column, ids in observed[node].indices.items():
                np.testing.assert_array_equal(samples[node].indices[column], ids)

            assert set(samples[node].values) == set(observed[node].values)

            for variable, values in observed[node].values.items():
                assert samples[node].values[variable].shape == (len(values), num_samples, values.shape[-1])
                assert isinstance(samples[node].values[variable], np.ndarray)

    assert set(samples) == latent_nodes


NETWORKS = {
    "flow_matching": FlowMatching,
    "diffusion": DiffusionModel,
    "consistency": lambda: ConsistencyModel(total_steps=10),
    "stable_consistency": StableConsistencyModel,
}


@pytest.mark.parametrize("simulator", ["two_level_simulator", "crossed_design_irt_simulator"])
@pytest.mark.parametrize("network", NETWORKS)
def test_fit_sample_networks(simulator, network, request):
    simulator = request.getfixturevalue(simulator)
    estimator = estimator_for(simulator, NETWORKS[network])
    estimator.compile()

    estimator.fit(simulator=simulator, batch_size=2, num_batches=1, epochs=1)

    observed = simulator.sample(3)
    assert_samples_match(estimator, observed, estimator.sample(num_samples=5, conditions=observed), 5)


def test_sample_tensor_format(two_level_simulator, two_level_estimator):
    two_level_estimator.build_from_data(two_level_estimator.to_batch(two_level_simulator.sample(2)))

    samples = two_level_estimator.sample(num_samples=5, conditions=two_level_simulator.sample(3), format="tensor")

    assert set(samples) == {"hyper_mean", "hyper_std", "local_mean", "shared_std"}
    assert samples["hyper_mean"].shape == (3, 5, 1)
    assert samples["local_mean"].shape == (3, 5, 6, 1)


def test_sample_tensor_format_raises_for_ragged_datasets(crossed_design_irt_simulator, crossed_design_irt_estimator):
    estimator = crossed_design_irt_estimator
    estimator.build_from_data(estimator.to_batch(crossed_design_irt_simulator.sample(2)))
    observed = crossed_design_irt_simulator.sample(3)

    with pytest.raises(ValueError, match="node 'students'"):
        estimator.sample(num_samples=5, conditions=observed, format="tensor")

    posterior = estimator.sample(num_samples=5, conditions=observed)
    num_students = np.sum(observed["students"].indices["dataset"] == 1)

    assert to_tensors(posterior, datasets=1)["student_ability"].shape == (1, 5, num_students, 1)


def test_save_and_load_flow_matching(tmp_path, two_level_simulator):
    estimator = estimator_for(two_level_simulator, FlowMatching)
    estimator.build_from_data(estimator.to_batch(two_level_simulator.sample(2)))

    estimator.save(tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert len(loaded.weights) == len(estimator.weights)
    for loaded_weight, weight in zip(loaded.weights, estimator.weights):
        np.testing.assert_allclose(keras.ops.convert_to_numpy(loaded_weight), keras.ops.convert_to_numpy(weight))


@pytest.mark.parametrize("network", ["consistency", "stable_consistency"])
def test_log_prob_raises_for_consistency_models(network, two_level_simulator):
    estimator = estimator_for(two_level_simulator, NETWORKS[network])
    estimator.build_from_data(estimator.to_batch(two_level_simulator.sample(2)))

    with pytest.raises(NotImplementedError, match="no tractable density"):
        estimator.log_prob(two_level_simulator.sample(2))


@pytest.mark.parametrize(("simulator", "estimator"), EXAMPLES)
def test_padding_preserves_loss(simulator, estimator, request):
    simulator = request.getfixturevalue(simulator)
    estimator = request.getfixturevalue(estimator)

    simulation = simulator.sample(5)
    batch = estimator.to_batch(simulation)
    estimator.build_from_data(batch)

    # fit converts the batches to tensors, so the padding weights are on the same device as the loss
    padded_batch = keras.tree.map_structure(keras.ops.convert_to_tensor, estimator.to_batch(simulation, padded=True))

    plain = estimator.compute_metrics(**batch, stage="validation")
    padded = estimator.compute_metrics(**padded_batch, stage="validation")

    for name, value in plain.items():
        np.testing.assert_allclose(
            keras.ops.convert_to_numpy(padded[name]), keras.ops.convert_to_numpy(value), rtol=1e-4, atol=1e-4
        )


def test_fit_with_validation_data(two_level_simulator, two_level_estimator):
    two_level_estimator.compile()

    history = two_level_estimator.fit(
        dataset=two_level_simulator.sample(4), validation_data=two_level_simulator.sample(3), batch_size=2, epochs=1
    )

    assert "val_loss" in history.history


def test_log_prob_sums_rows_per_dataset(two_level_simulator, two_level_estimator):
    simulation = two_level_simulator.sample(5)
    two_level_estimator.build_from_data(two_level_estimator.to_batch(simulation))

    log_prob = two_level_estimator.log_prob(simulation)

    np.testing.assert_allclose(
        two_level_estimator.log_prob(concatenate([simulation] * 2)), np.tile(log_prob, 2), atol=1e-4
    )


def test_log_prob_with_adapter(two_level_simulator):
    adapter = Adapter().standardize(include="hyper_mean", mean=0.0, std=2.0)
    estimator = MultilevelEstimator(
        two_level_simulator.graph.expand().invert(),
        adapter=adapter,
        inference_networks=[CouplingFlow(), CouplingFlow()],
    )

    simulation = two_level_simulator.sample(2)
    estimator.build_from_data(estimator.to_batch(simulation))

    log_prob = estimator.log_prob(simulation)
    assert log_prob.shape == (2,)
    assert np.all(np.isfinite(log_prob))


@pytest.mark.parametrize(("simulator", "estimator"), EXAMPLES)
def test_serialization(simulator, estimator, request):
    estimator = request.getfixturevalue(estimator)

    restored = MultilevelEstimator.from_config(estimator.get_config())

    assert restored.networks == estimator.networks
    assert restored.condition_nodes == estimator.condition_nodes


def test_save_and_load(tmp_path, two_data_nodes_simulator, two_data_nodes_estimator):
    estimator = two_data_nodes_estimator
    estimator.build_from_data(estimator.to_batch(two_data_nodes_simulator.sample(2)))

    estimator.save(tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert loaded.networks == estimator.networks
    assert loaded.condition_nodes == estimator.condition_nodes
    assert loaded.variable_dims == estimator.variable_dims

    simulation = two_data_nodes_simulator.sample(2)
    np.testing.assert_allclose(loaded.log_prob(simulation), estimator.log_prob(simulation), rtol=1e-5)


def test_defaults(two_data_nodes_estimator):
    estimator = two_data_nodes_estimator

    assert isinstance(estimator.adapter, Adapter)
    assert estimator.data_nodes == ["w", "y"]

    # one encoder per network and data node it conditions on
    for network_encoders in estimator.encoders:
        assert sorted(network_encoders) == ["w", "y"]
        for encoder in network_encoders.values():
            assert isinstance(encoder, MultilevelEncoder)


def test_non_amortizable_networks_are_sequential(crossed_design_irt_estimator):
    assert crossed_design_irt_estimator.sequential == ["questions"]


def test_sequential_network_rejects_missing_target_probability(crossed_design_irt_simulator):
    def network():
        return CouplingFlow(mask_handler=MaskHandler(fixed_target_prob=0.5, missing_target_prob=0.5))

    with pytest.raises(NotImplementedError, match="cannot marginalize target features"):
        estimator_for(crossed_design_irt_simulator, network)
