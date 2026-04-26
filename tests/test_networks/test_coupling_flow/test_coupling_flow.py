import keras
import numpy as np

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal, assert_allclose


def test_build(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not coupling_flow.built
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)
    assert coupling_flow.built
    assert coupling_flow.variables


def test_forward_output_shape(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    z = coupling_flow(random_samples, conditions=random_conditions)
    assert keras.ops.shape(z) == keras.ops.shape(random_samples)


def test_forward_density_output_shape(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    z, log_density = coupling_flow(random_samples, conditions=random_conditions, density=True)
    assert keras.ops.shape(z) == keras.ops.shape(random_samples)
    assert keras.ops.shape(log_density) == (keras.ops.shape(random_samples)[0],)


def test_inverse_output_shape(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x = coupling_flow(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)


def test_inverse_density_output_shape(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x, log_density = coupling_flow(z, conditions=random_conditions, inverse=True, density=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)
    assert keras.ops.shape(log_density) == (keras.ops.shape(random_samples)[0],)


def test_variable_batch_size(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs,) + keras.ops.shape(random_samples)[1:])
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = coupling_flow(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


def test_cycle_consistency(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    z, fwd_log_density = coupling_flow(random_samples, conditions=random_conditions, density=True)
    x_reconstructed, inv_log_density = coupling_flow(z, conditions=random_conditions, inverse=True, density=True)

    assert_allclose(random_samples, x_reconstructed, atol=1e-5, rtol=1e-5)
    assert_allclose(fwd_log_density, inv_log_density, atol=1e-3, rtol=1e-3)


def test_serialize_deserialize(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(coupling_flow)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "coupling_flow.keras"
    keras.saving.save_model(coupling_flow, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(coupling_flow, loaded)


def test_save_load_output_unchanged(tmp_path, random_samples, random_conditions):
    """Loaded model produces the same output as the original."""
    from bayesflow.networks import CouplingFlow

    model = CouplingFlow(depth=2, subnet_kwargs=dict(widths=(8, 8)))
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    original_out = model(z, conditions=random_conditions, inverse=True)

    path = tmp_path / "cf_output_check.keras"
    keras.saving.save_model(model, path)
    loaded = keras.saving.load_model(path)

    loaded_out = loaded(z, conditions=random_conditions, inverse=True)
    assert_allclose(original_out, loaded_out, atol=1e-5, rtol=1e-5)


def test_compute_metrics(coupling_flow, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    coupling_flow.build(xz_shape, conditions_shape=cond_shape)

    metrics = coupling_flow.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
