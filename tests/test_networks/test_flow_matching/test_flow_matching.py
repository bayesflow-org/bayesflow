import keras
import numpy as np

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_allclose, skip_torch_linalg_on_mps


class ConstantVelocitySubnet(keras.Layer):
    def __init__(self, velocity):
        super().__init__()
        self.velocity = keras.ops.convert_to_tensor(velocity)

    def call(self, inputs):
        return keras.ops.ones_like(inputs[0]) * self.velocity

    def compute_output_shape(self, input_shape):
        return input_shape[0]


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


def test_training_path_runs_from_data_to_noise(monkeypatch):
    """Regression test for the t=0 data, t=1 noise convention and its velocity sign."""
    from bayesflow.networks import FlowMatching

    data = keras.ops.convert_to_tensor([[2.0, -1.0], [4.0, 3.0], [-2.0, 5.0]])
    noise = keras.ops.convert_to_tensor([[8.0, 7.0], [-2.0, 9.0], [6.0, 1.0]])
    time = keras.ops.convert_to_tensor([0.0, 0.25, 1.0])
    captured = {}

    def loss_fn(target, prediction):
        captured["target_velocity"] = target
        return keras.ops.mean(keras.ops.square(target - prediction), axis=-1)

    model = FlowMatching(loss_fn=loss_fn, subnet_kwargs={"widths": (8, 8)}, time_power_law_alpha=0.0)
    monkeypatch.setattr(model.base_distribution, "sample", lambda batch_shape, seed=None: noise)
    monkeypatch.setattr(keras.random, "uniform", lambda shape, seed=None: 1.0 - time)

    def velocity(xz, *, time, conditions, training, masks):
        captured["path"] = xz
        captured["time"] = time
        return keras.ops.zeros_like(xz)

    monkeypatch.setattr(model, "_velocity", velocity)
    model.compute_metrics(data)

    expanded_time = keras.ops.expand_dims(time, axis=-1)
    expected_path = (1.0 - expanded_time) * data + expanded_time * noise
    assert_allclose(captured["time"], expanded_time)
    assert_allclose(captured["path"], expected_path)
    assert_allclose(captured["path"][0], data[0])
    assert_allclose(captured["path"][-1], noise[-1])
    assert_allclose(captured["target_velocity"], noise - data)


def test_forward_and_inverse_follow_data_to_noise_time_direction():
    """Forward integrates 0 -> 1; inverse sampling integrates 1 -> 0."""
    from bayesflow.networks import FlowMatching

    velocity = keras.ops.convert_to_tensor([2.0, -0.5])
    data = keras.ops.convert_to_tensor([[1.0, -2.0], [3.0, 0.5]])
    noise = data + velocity
    model = FlowMatching(
        subnet=ConstantVelocitySubnet(velocity),
        integrate_kwargs={"method": "euler", "steps": 1},
    )
    model.build(keras.ops.shape(data))

    assert_allclose(model(data), noise)
    assert_allclose(model(noise, inverse=True), data)


def test_compute_metrics(flow_matching, random_samples, random_conditions):
    if flow_matching.use_optimal_transport and random_conditions is not None:
        skip_torch_linalg_on_mps()

    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    flow_matching.build(xz_shape, conditions_shape=cond_shape)

    metrics = flow_matching.compute_metrics(random_samples, conditions=random_conditions)
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
