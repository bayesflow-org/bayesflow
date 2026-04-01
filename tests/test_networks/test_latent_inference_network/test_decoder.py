import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal


def test_build(decoder, random_samples, latent_dim):
    input_shape = keras.ops.shape(random_samples)
    output_dim = input_shape[-1]

    decoder.output_dim = output_dim
    latent_shape = (input_shape[0], latent_dim)

    assert not decoder.built
    decoder.build(latent_shape)
    assert decoder.built
    assert decoder.variables


def test_build_raises_without_output_dim(decoder, latent_dim):
    latent_shape = (2, latent_dim)

    with pytest.raises(ValueError, match="output_dim must be set"):
        decoder.build(latent_shape)


def test_output_shape(decoder, random_samples, latent_dim):
    input_shape = keras.ops.shape(random_samples)
    output_dim = input_shape[-1]

    decoder.output_dim = output_dim
    latent_shape = (input_shape[0], latent_dim)
    decoder.build(latent_shape)

    z = keras.random.normal(latent_shape)
    x_recon = decoder(z)

    assert keras.ops.shape(x_recon) == (input_shape[0], output_dim)


def test_serialize_deserialize(decoder, random_samples, latent_dim):
    input_shape = keras.ops.shape(random_samples)
    decoder.output_dim = input_shape[-1]
    latent_shape = (input_shape[0], latent_dim)
    decoder.build(latent_shape)

    serialized = serialize(decoder)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, decoder, random_samples, latent_dim):
    input_shape = keras.ops.shape(random_samples)
    decoder.output_dim = input_shape[-1]
    latent_shape = (input_shape[0], latent_dim)
    decoder.build(latent_shape)

    path = tmp_path / "decoder.keras"
    keras.saving.save_model(decoder, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(decoder, loaded)
