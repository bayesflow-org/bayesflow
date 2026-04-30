import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal


def test_build(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not lin_with_diffusion.built
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)
    assert lin_with_diffusion.built
    assert lin_with_diffusion.variables


def test_forward_output_shape(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    z = lin_with_diffusion(random_samples, conditions=random_conditions)
    assert keras.ops.shape(z) == (xz_shape[0], lin_with_diffusion.latent_dim)


def test_inverse_output_shape(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal((xz_shape[0], lin_with_diffusion.latent_dim))
    out = lin_with_diffusion(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(out) == xz_shape


def test_encode_decode(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    z, mean, log_var = lin_with_diffusion.encode(random_samples)
    assert keras.ops.shape(z) == (xz_shape[0], lin_with_diffusion.latent_dim)
    assert keras.ops.shape(mean) == (xz_shape[0], lin_with_diffusion.latent_dim)
    assert keras.ops.shape(log_var) == (xz_shape[0], lin_with_diffusion.latent_dim)

    x_recon = lin_with_diffusion.decode(z)
    assert keras.ops.shape(x_recon) == xz_shape


def test_forward_density_raises(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    with pytest.raises(NotImplementedError, match="Exact density computation"):
        lin_with_diffusion(random_samples, conditions=random_conditions, density=True)


def test_inverse_density_raises(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal((xz_shape[0], lin_with_diffusion.latent_dim))
    with pytest.raises(NotImplementedError, match="Exact density computation"):
        lin_with_diffusion(z, conditions=random_conditions, inverse=True, density=True)


def test_compute_metrics(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    metrics = lin_with_diffusion.compute_metrics(random_samples, conditions=random_conditions)

    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_components(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    metrics = lin_with_diffusion.compute_metrics(random_samples, conditions=random_conditions)

    expected_keys = {"loss", "reconstruction_loss", "kl_loss", "inference_loss", "warmup_weight"}
    assert expected_keys <= set(metrics.keys()), f"Missing keys: {expected_keys - set(metrics.keys())}"

    for key in expected_keys:
        value = keras.ops.convert_to_numpy(metrics[key])
        assert np.isfinite(value), f"{key} is not finite: {value}"


def test_warmup_weight(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    warmup = lin_with_diffusion._compute_warmup_weight()
    warmup_np = keras.ops.convert_to_numpy(warmup)
    assert warmup_np >= 0.0
    assert warmup_np <= 1.0


def test_variable_batch_size(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs, lin_with_diffusion.latent_dim))
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = lin_with_diffusion(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


def test_serialize_deserialize(lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(lin_with_diffusion)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, lin_with_diffusion, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    lin_with_diffusion.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "lin.keras"
    keras.saving.save_model(lin_with_diffusion, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(lin_with_diffusion, loaded)


def test_with_flow_matching(lin_with_flow_matching, random_samples, random_conditions):
    model = lin_with_flow_matching
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
