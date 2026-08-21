import keras
import numpy as np

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_allclose


# ---- Configuration ---------------------------------------------------------


def test_build_with_custom_integrate_kwargs():
    from bayesflow.networks import FlowMatching

    model = FlowMatching(
        subnet_kwargs=dict(widths=(8, 8)),
        integrate_kwargs=dict(method="euler", steps=10),
    )
    model.build((2, 3), conditions_shape=(2, 3))
    assert model.built
    assert model.integrate_kwargs["method"] == "euler"
    assert model.integrate_kwargs["steps"] == 10


# ---- compute_metrics (training path of each variant) ------------------------


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


def test_adaptive_solver_sample_and_density():
    """The instance default is an adaptive tsit5 solver for sampling and density.
    The adaptive density must match a high-accuracy fixed-step reference, which
    in turn is verified against a numerical jacobian in
    test_inference_networks.py::test_density_numerically."""
    from bayesflow.networks import FlowMatching

    model = FlowMatching(subnet_kwargs=dict(widths=(8, 8)))
    conditions = keras.random.normal((2, 3))
    model.build((2, 3), conditions_shape=(2, 3))

    # sampling with the default adaptive solver
    z = keras.random.normal((2, 3))
    x = model(z, conditions=conditions, inverse=True)
    assert keras.ops.shape(x) == (2, 3)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(x)))

    # density with the adaptive solver, in both directions
    x_adaptive, ld_inv_adaptive = model(z, conditions=conditions, inverse=True, density=True)
    z_adaptive, ld_fwd_adaptive = model(x_adaptive, conditions=conditions, density=True)

    # high-accuracy fixed-step reference
    model.integrate_kwargs.update({"steps": 150})
    x_fixed, ld_inv_fixed = model(z, conditions=conditions, inverse=True, density=True)
    z_fixed, ld_fwd_fixed = model(x_adaptive, conditions=conditions, density=True)

    assert_allclose(x_adaptive, x_fixed, atol=1e-3, rtol=1e-3)
    assert_allclose(ld_inv_adaptive, ld_inv_fixed, atol=1e-3, rtol=1e-3)
    assert_allclose(z_adaptive, z_fixed, atol=1e-3, rtol=1e-3)
    assert_allclose(ld_fwd_adaptive, ld_fwd_fixed, atol=1e-3, rtol=1e-3)


# ---- Serialization ---------------------------------------------------------


def test_serialize_deserialize(flow_matching, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(flow_matching)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_load_output_unchanged(tmp_path, random_samples, random_conditions):
    """Loaded model produces the same output as the original."""
    from bayesflow.networks import FlowMatching

    # a few fixed euler steps keep sampling cheap and deterministic
    model = FlowMatching(subnet_kwargs=dict(widths=(8, 8)), integrate_kwargs=dict(method="euler", steps=8))
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
