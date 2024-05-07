
import keras
import numpy as np
import pytest

import bayesflow.experimental as bf


# TODO: @Chase outdated (local/global context, WIP)


@pytest.fixture(scope="module")
def batch_shape():
    return (16,)


@pytest.fixture(scope="module")
def two_moons_conditions():
    @bf.distribution
    def prior():
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        return {"r": r, "alpha": alpha}

    return prior


@pytest.fixture(scope="module")
def two_moons_prior():
    @bf.distribution
    def prior():
        theta = np.random.uniform(-1.0, 1.0, size=2)
        return {"theta": theta}

    return prior


def test_unconditional_prior(batch_shape):
    @bf.distribution
    def unconditional_prior():
        return {"theta": np.random.normal(size=2)}

    parameters = unconditional_prior.sample(batch_shape)

    assert isinstance(parameters, dict)
    assert list(parameters.keys()) == ["theta"]
    assert keras.ops.is_tensor(parameters["theta"])
    assert keras.ops.shape(parameters["theta"]) == batch_shape + (2,)


def test_conditional_prior(batch_shape, two_moons_conditions):
    @bf.distribution
    def conditional_prior(conditions):
        assert isinstance(conditions, dict)
        assert "r" in conditions
        assert "alpha" in conditions
        assert keras.ops.is_tensor(conditions["r"])
        assert keras.ops.is_tensor(conditions["alpha"])
        assert keras.ops.shape(conditions["r"]) == (1,)
        assert keras.ops.shape(conditions["alpha"]) == (1,)

        return {"theta": np.random.normal(size=2)}

    conditions = two_moons_conditions(batch_shape)
    parameters = conditional_prior.sample(batch_shape, **conditions)

    assert isinstance(parameters, dict)
    assert list(parameters.keys()) == ["theta"]
    assert keras.ops.is_tensor(parameters["theta"])
    assert keras.ops.shape(parameters["theta"]) == batch_shape + (2,)


def test_conditional_prior_with_kwargs(batch_shape, two_moons_conditions):
    @bf.distribution
    def conditional_prior(r, alpha):
        assert keras.ops.is_tensor(r)
        assert keras.ops.is_tensor(alpha)
        assert keras.ops.shape(r) == (1,)
        assert keras.ops.shape(alpha) == (1,)
        return {"theta": np.random.normal(size=2)}

    conditions = two_moons_conditions(batch_shape)
    parameters = conditional_prior.sample(batch_shape, **conditions)

    assert isinstance(parameters, dict)
    assert list(parameters.keys()) == ["theta"]
    assert keras.ops.is_tensor(parameters["theta"])
    assert keras.ops.shape(parameters["theta"]) == batch_shape + (2,)


def test_randomness_of_unbatched_prior(batch_shape):
    @bf.distribution(is_batched=False)
    def prior():
        return {"x": keras.random.normal(shape=(1,))}

    sample_dict = prior.sample(batch_shape)
    sample_arr = sample_dict['x']

    # check that individual values in the batch are not all same
    assert not keras.ops.all(keras.ops.isclose(sample_arr, sample_arr[0]))


def test_randomness_of_batched_prior(batch_shape):
    @bf.distribution(is_batched=True)
    def prior(batch_shape):
        return {"x": keras.random.normal(shape=batch_shape + (1,))}

    sample_dict = prior.sample(batch_shape)
    sample_arr = sample_dict['x']

    # check that individual values in the batch are not all same
    assert not keras.ops.all(keras.ops.isclose(sample_arr, sample_arr[0]))