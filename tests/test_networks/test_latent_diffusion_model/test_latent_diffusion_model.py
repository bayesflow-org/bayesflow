import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal


def test_build(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not latent_diffusion_model.built
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)
    assert latent_diffusion_model.built
    assert latent_diffusion_model.variables


def test_compute_metrics(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = latent_diffusion_model.compute_metrics(random_samples, conditions=random_conditions)

    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_serialize_deserialize(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(latent_diffusion_model)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "ldm.keras"
    keras.saving.save_model(latent_diffusion_model, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(latent_diffusion_model, loaded)


def test_with_flow_matching(latent_diffusion_model_with_flow_matching, random_samples, random_conditions):
    model = latent_diffusion_model_with_flow_matching
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    model.build(xz_shape, conditions_shape=cond_shape)

    metrics = model.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"

    z = keras.random.normal((xz_shape[0], model.latent_dim))
    out = model(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(out) == xz_shape
