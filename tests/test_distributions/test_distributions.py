import keras
from keras.saving import (
    serialize_keras_object as serialize,
    deserialize_keras_object as deserialize,
)
import pytest


def test_sample_output_shape(distribution, shape):
    distribution.build(shape)
    samples = distribution.sample(shape[:1])
    assert keras.ops.shape(samples) == shape


def test_log_prob_output_shape(distribution, random_samples):
    distribution.build(keras.ops.shape(random_samples))
    log_prob = distribution.log_prob(random_samples)
    assert keras.ops.shape(log_prob) == keras.ops.shape(random_samples)[:1]


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
