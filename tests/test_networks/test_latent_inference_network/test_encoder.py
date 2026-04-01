import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal


def test_build(encoder, random_samples):
    input_shape = keras.ops.shape(random_samples)

    assert not encoder.built
    encoder.build(input_shape)
    assert encoder.built
    assert encoder.variables


def test_output_shape(encoder, random_samples):
    input_shape = keras.ops.shape(random_samples)
    encoder.build(input_shape)

    z, mean, log_var = encoder(random_samples)

    assert keras.ops.shape(z) == (input_shape[0], encoder.latent_dim)
    assert keras.ops.shape(mean) == (input_shape[0], encoder.latent_dim)
    assert keras.ops.shape(log_var) == (input_shape[0], encoder.latent_dim)


def test_auto_latent_dim(encoder_auto, random_samples):
    input_shape = keras.ops.shape(random_samples)
    encoder_auto.build(input_shape)

    expected_latent_dim = max(2, input_shape[-1] // 2)
    assert encoder_auto.latent_dim == expected_latent_dim

    z, mean, log_var = encoder_auto(random_samples)
    assert keras.ops.shape(z) == (input_shape[0], expected_latent_dim)


def test_reparameterization(encoder, random_samples):
    input_shape = keras.ops.shape(random_samples)
    encoder.build(input_shape)

    z1, mean1, log_var1 = encoder(random_samples)
    z2, mean2, log_var2 = encoder(random_samples)

    # Means and log_vars should be deterministic
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(mean1),
        keras.ops.convert_to_numpy(mean2),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(log_var1),
        keras.ops.convert_to_numpy(log_var2),
        atol=1e-5,
    )

    # Sampled z should differ due to stochastic reparameterization
    z1_np = keras.ops.convert_to_numpy(z1)
    z2_np = keras.ops.convert_to_numpy(z2)
    assert not np.allclose(z1_np, z2_np, atol=1e-5), "z samples should be stochastic"


def test_serialize_deserialize(encoder, random_samples):
    input_shape = keras.ops.shape(random_samples)
    encoder.build(input_shape)

    serialized = serialize(encoder)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, encoder, random_samples):
    input_shape = keras.ops.shape(random_samples)
    encoder.build(input_shape)

    path = tmp_path / "encoder.keras"
    keras.saving.save_model(encoder, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(encoder, loaded)
