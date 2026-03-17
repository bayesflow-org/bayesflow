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


def test_forward_output_shape(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z = latent_diffusion_model(random_samples, conditions=random_conditions)
    assert keras.ops.shape(z) == (xz_shape[0], latent_diffusion_model.latent_dim)


def test_inverse_output_shape(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal((xz_shape[0], latent_diffusion_model.latent_dim))
    out = latent_diffusion_model(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(out) == xz_shape


def test_encode_decode(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z, mean, log_var = latent_diffusion_model.encode(random_samples)
    assert keras.ops.shape(z) == (xz_shape[0], latent_diffusion_model.latent_dim)
    assert keras.ops.shape(mean) == (xz_shape[0], latent_diffusion_model.latent_dim)
    assert keras.ops.shape(log_var) == (xz_shape[0], latent_diffusion_model.latent_dim)

    x_recon = latent_diffusion_model.decode(z)
    assert keras.ops.shape(x_recon) == xz_shape


def test_forward_density_raises(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    with pytest.raises(NotImplementedError, match="Exact density computation"):
        latent_diffusion_model(random_samples, conditions=random_conditions, density=True)


def test_inverse_density_raises(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal((xz_shape[0], latent_diffusion_model.latent_dim))
    with pytest.raises(NotImplementedError, match="Exact density computation"):
        latent_diffusion_model(z, conditions=random_conditions, inverse=True, density=True)


def test_compute_metrics(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = latent_diffusion_model.compute_metrics(random_samples, conditions=random_conditions)

    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_components(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = latent_diffusion_model.compute_metrics(random_samples, conditions=random_conditions)

    expected_keys = {"loss", "reconstruction_loss", "kl_loss", "inference_loss", "warmup_weight"}
    assert expected_keys <= set(metrics.keys()), f"Missing keys: {expected_keys - set(metrics.keys())}"

    for key in expected_keys:
        value = keras.ops.convert_to_numpy(metrics[key])
        assert np.isfinite(value), f"{key} is not finite: {value}"


def test_warmup_weight(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    # Initially, warmup weight should be 0 (step 0 / warmup_steps)
    warmup = latent_diffusion_model._compute_warmup_weight()
    warmup_np = keras.ops.convert_to_numpy(warmup)
    assert warmup_np >= 0.0
    assert warmup_np <= 1.0


def test_variable_batch_size(latent_diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    latent_diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs, latent_diffusion_model.latent_dim))
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = latent_diffusion_model(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


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

    # Test compute_metrics works
    metrics = model.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"

    # Test inverse (sampling) works
    z = keras.random.normal((xz_shape[0], model.latent_dim))
    out = model(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(out) == xz_shape
