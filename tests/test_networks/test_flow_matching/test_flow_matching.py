import keras
import numpy as np

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal, assert_allclose


# ---- Build -----------------------------------------------------------------


def test_build(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not flow_matching.built
    flow_matching.build(xz_shape, conditions_shape=cond_shape)
    assert flow_matching.built
    assert flow_matching.variables


def test_build_with_custom_integrate_kwargs(random_samples, random_conditions):
    from bayesflow.networks import FlowMatching

    model = FlowMatching(
        subnet_kwargs=dict(widths=(8, 8)),
        integrate_kwargs=dict(method="euler", steps=10),
    )
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    model.build(xz_shape, conditions_shape=cond_shape)
    assert model.built
    assert model.integrate_kwargs["method"] == "euler"
    assert model.integrate_kwargs["steps"] == 10


# ---- Output shapes ---------------------------------------------------------


def test_forward_output_shape(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    z = flow_matching(random_samples, conditions=random_conditions)
    assert keras.ops.shape(z) == keras.ops.shape(random_samples)


def test_forward_density_output_shape(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    z, log_density = flow_matching(random_samples, conditions=random_conditions, density=True)
    assert keras.ops.shape(z) == keras.ops.shape(random_samples)
    assert keras.ops.shape(log_density) == (keras.ops.shape(random_samples)[0],)


def test_inverse_output_shape(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x = flow_matching(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)


def test_inverse_density_output_shape(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x, log_density = flow_matching(z, conditions=random_conditions, inverse=True, density=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)
    assert keras.ops.shape(log_density) == (keras.ops.shape(random_samples)[0],)


# ---- Variable batch size ---------------------------------------------------


def test_variable_batch_size(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs,) + keras.ops.shape(random_samples)[1:])
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = flow_matching(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


# ---- Cycle consistency (forward ∘ inverse ≈ identity) ----------------------


def test_cycle_consistency(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    z, fwd_log_density = flow_matching(random_samples, conditions=random_conditions, density=True)
    x_reconstructed, inv_log_density = flow_matching(z, conditions=random_conditions, inverse=True, density=True)

    assert_allclose(random_samples, x_reconstructed, atol=1e-3, rtol=1e-3)
    assert_allclose(fwd_log_density, inv_log_density, atol=1e-3, rtol=1e-3)


# ---- Serialization ---------------------------------------------------------


def test_serialize_deserialize(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(flow_matching)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "flow_matching.keras"
    keras.saving.save_model(flow_matching, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(flow_matching, loaded)


def test_save_load_output_unchanged(tmp_path, random_samples, random_conditions):
    """Loaded model produces the same output as the original."""
    from bayesflow.networks import FlowMatching

    model = FlowMatching(subnet_kwargs=dict(widths=(8, 8)))
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    original_out = model(z, conditions=random_conditions, inverse=True)

    path = tmp_path / "fm_output_check.keras"
    keras.saving.save_model(model, path)
    loaded = keras.saving.load_model(path)

    loaded_out = loaded(z, conditions=random_conditions, inverse=True)
    assert_allclose(original_out, loaded_out, atol=1e-5, rtol=1e-5)


# ---- compute_metrics -------------------------------------------------------


def test_compute_metrics(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    metrics = flow_matching.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_with_masking(flow_matching_with_masking, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching_with_masking.build(xz_shape, conditions_shape=cond_shape)

    metrics = flow_matching_with_masking.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
