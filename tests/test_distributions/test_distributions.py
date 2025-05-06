import pytest

import numpy as np
from scipy.stats import norm, multivariate_t

import keras

from bayesflow.distributions import DiagonalNormal, DiagonalStudentT, Mixture
from bayesflow.utils.serialization import serialize, deserialize


def test_sample_output_shape(distribution, shape):
    distribution.build(shape)
    samples = distribution.sample(shape[:1])
    assert keras.ops.shape(samples) == shape


def test_log_prob_output_shape(distribution, random_samples):
    distribution.build(keras.ops.shape(random_samples))
    log_prob = distribution.log_prob(random_samples)
    assert keras.ops.shape(log_prob) == keras.ops.shape(random_samples)[:1]


def test_log_prob_correctness(distribution, random_samples):
    distribution.build(keras.ops.shape(random_samples))
    log_prob = distribution.log_prob(random_samples)
    log_prob = keras.ops.convert_to_numpy(log_prob)
    random_samples = keras.ops.convert_to_numpy(random_samples)

    if isinstance(distribution, DiagonalNormal):
        loc = keras.ops.convert_to_numpy(distribution.mean)
        scale = keras.ops.convert_to_numpy(distribution.std)
        log_prob_scipy = norm(loc=loc, scale=scale).logpdf(random_samples)
        log_prob_scipy = log_prob_scipy.sum(axis=-1)

    elif isinstance(distribution, DiagonalStudentT):
        loc = keras.ops.convert_to_numpy(distribution.loc)
        scale = keras.ops.convert_to_numpy(distribution.scale)
        df = distribution.df
        log_prob_scipy = multivariate_t(loc=loc, shape=np.diag(scale**2), df=df).logpdf(random_samples)

    elif isinstance(distribution, Mixture):
        loc = keras.ops.convert_to_numpy(distribution.distributions[0].mean)
        scale = keras.ops.convert_to_numpy(distribution.distributions[0].std)
        log_prob_norm_scipy = norm(loc=loc, scale=scale).logpdf(random_samples)
        log_prob_norm_scipy = log_prob_norm_scipy.sum(axis=-1)

        loc = keras.ops.convert_to_numpy(distribution.distributions[1].loc)
        scale = keras.ops.convert_to_numpy(distribution.distributions[1].scale)
        df = distribution.distributions[1].df
        log_prob_t_scipy = multivariate_t(loc=loc, shape=np.diag(scale**2), df=df).logpdf(random_samples)
        log_prob_scipy = np.log(0.5 * np.exp(log_prob_norm_scipy) + 0.5 * np.exp(log_prob_t_scipy))

    else:
        raise RuntimeError("distribution must be in '[DiagonalNormal, DiagonalStudentT, Mixture]'")

    assert np.allclose(log_prob, log_prob_scipy)


@pytest.mark.parametrize("automatic", [True, False])
def test_build(automatic, distribution, random_samples):
    assert distribution.built is False

    if automatic:
        distribution(random_samples)
    else:
        distribution.build(keras.ops.shape(random_samples))

    assert distribution.built is True


def test_serialize_deserialize(distribution, random_samples):
    distribution.build(keras.ops.shape(random_samples))

    serialized = serialize(distribution)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized
