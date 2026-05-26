import pytest
import keras
import numpy as np


def test_two_moons(two_moons_simulator, batch_size):
    samples = two_moons_simulator.sample((batch_size,))

    assert isinstance(samples, dict)
    assert list(samples.keys()) == ["parameters", "observables"]
    assert all(isinstance(value, np.ndarray) for value in samples.values())

    assert samples["parameters"].shape == (batch_size, 2)
    assert samples["observables"].shape == (batch_size, 2)


def test_gaussian_linear(gaussian_linear_simulator, batch_size):
    samples = gaussian_linear_simulator.sample((batch_size,))

    # test n_obs respected if applicable
    if hasattr(gaussian_linear_simulator, "n_obs") and isinstance(gaussian_linear_simulator.n_obs, int):
        assert samples["observables"].shape[1] == gaussian_linear_simulator.n_obs


def test_sample(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert isinstance(value, np.ndarray)

        # test shape
        assert value.shape[0] == batch_size

        # test batch randomness
        assert not np.allclose(value, value[0])


def test_sample_batched(simulator, batch_size):
    sample_size = 2
    samples = simulator.sample_batched((batch_size,), sample_size=sample_size)

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert isinstance(value, np.ndarray)

        # test shape (sample_batched rounds up to complete batches)
        assert value.shape[0] == int(np.ceil(batch_size / sample_size)) * sample_size

        # test batch randomness
        assert not np.allclose(value, value[0])


def test_fixed_sample(composite_gaussian, batch_size, fixed_n, fixed_mu):
    samples = composite_gaussian.sample((batch_size,), n=fixed_n, mu=fixed_mu)

    assert samples["n"] == fixed_n
    assert samples["mu"].shape == (batch_size, 1)
    assert np.all(samples["mu"] == fixed_mu)
    assert samples["y"].shape == (batch_size, fixed_n)


def test_multimodel_sample(multimodel, batch_size):
    samples = multimodel.sample(batch_size)

    assert set(samples) == {"n", "mu", "y", "model_indices"}
    assert samples["mu"].shape == (batch_size, 1)
    assert samples["y"].shape == (batch_size, samples["n"])


def test_multimodel_key_conflicts_sample(multimodel_key_conflicts, batch_size):
    if multimodel_key_conflicts.key_conflicts == "drop":
        samples = multimodel_key_conflicts.sample(batch_size)
        assert set(samples) == {"x", "model_indices"}
    elif multimodel_key_conflicts.key_conflicts == "fill":
        samples = multimodel_key_conflicts.sample(batch_size)
        assert set(samples) == {"x", "model_indices", "c", "w"}
        assert np.sum(np.isnan(samples["c"])) + np.sum(np.isnan(samples["w"])) == batch_size
    elif multimodel_key_conflicts.key_conflicts == "error":
        with pytest.raises(ValueError):
            samples = multimodel_key_conflicts.sample(batch_size)


def test_multimodel_single_batch(multimodel_single_batch, batch_size):
    samples = multimodel_single_batch.sample(batch_size)
    # all samples in the batch come from the same model
    assert {"mu", "y", "model_indices"}.issubset(set(samples))
    assert samples["model_indices"].shape == (batch_size, 2)
    # every row of model_indices must be identical (single model for whole batch)
    assert np.all(samples["model_indices"] == samples["model_indices"][0])


def test_multimodel_p_argument(batch_size):
    from bayesflow.simulators import make_simulator, ModelComparisonSimulator

    def prior_0():
        return dict(mu=0.0)

    def prior_1():
        return dict(mu=np.random.standard_normal())

    def likelihood(mu):
        return dict(y=np.random.normal(mu, 1, 4))

    sim0 = make_simulator([prior_0, likelihood])
    sim1 = make_simulator([prior_1, likelihood])

    # valid probabilities must sum to 1
    simulator = ModelComparisonSimulator(simulators=[sim0, sim1], p=[0.3, 0.7])
    samples = simulator.sample(batch_size)
    assert set(samples) >= {"model_indices", "y"}

    # invalid probabilities should raise
    with pytest.raises(ValueError, match="sum to 1"):
        ModelComparisonSimulator(simulators=[sim0, sim1], p=[0.3, 0.3])

    # p and logits are mutually exclusive
    with pytest.raises(ValueError, match="conflicting"):
        ModelComparisonSimulator(simulators=[sim0, sim1], p=[0.5, 0.5], logits=[0.0, 0.0])
