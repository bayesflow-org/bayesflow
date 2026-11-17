import keras
import numpy as np
import pytest

import bayesflow.experimental.mean_flow.mean_flow as mean_flow_module
from bayesflow.experimental import MeanFlow
from bayesflow.masking import InferenceMasks
from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_allclose, skip_torch_linalg_on_mps

from .conftest import build, make_mean_flow, max_abs_diff


@pytest.mark.parametrize("steps", [1, 4])
def test_make_time_schedule(steps):
    times = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(steps))
    assert times.shape == (steps + 1,)
    np.testing.assert_allclose(times[[0, -1]], [1.0, 0.0], atol=1e-6)
    assert np.all(np.diff(times) < 0.0)


def test_default_time_power_law_sampling(monkeypatch):
    draws = iter(
        (
            keras.ops.convert_to_tensor([0.0, 0.25, 1.0]),
            keras.ops.convert_to_tensor([1.0, 0.25, 0.0]),
        )
    )
    monkeypatch.setattr(mean_flow_module.keras.random, "uniform", lambda *args, **kwargs: next(draws))

    network = MeanFlow(subnet_kwargs=dict(widths=(8, 8)))
    s, t = network._sample_times(keras.ops.zeros((3, 2)))

    assert network.time_power_law_alpha == 0.5
    assert_allclose(s, keras.ops.convert_to_tensor([[0.0], [0.125], [1.0]]))
    assert_allclose(t, keras.ops.convert_to_tensor([[1.0], [0.125], [0.0]]))


def test_flow_map_is_average_velocity_step(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions)
    u = network._average_velocity(random_samples, 1.0, 0.25, random_conditions, False, InferenceMasks())
    mapped = network._flow_map(random_samples, 1.0, 0.25, random_conditions, False, InferenceMasks())
    assert_allclose(mapped, random_samples + (0.25 - 1.0) * u, atol=1e-6)


def test_compute_metrics(mean_flow, random_samples, random_conditions):
    if mean_flow.use_optimal_transport and random_conditions is not None:
        skip_torch_linalg_on_mps()

    build(mean_flow, random_samples, random_conditions)
    metrics = mean_flow.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_with_sample_weight(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions)
    sample_weight = keras.random.uniform(keras.ops.shape(random_samples)[:1])
    metrics = network.compute_metrics(random_samples, conditions=random_conditions, sample_weight=sample_weight)
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_optimal_transport_reorders_sample_weight(monkeypatch, random_samples, random_conditions):
    assignments = keras.ops.arange(keras.ops.shape(random_samples)[0] - 1, -1, -1)

    def fake_optimal_transport(z, x, conditions=None, return_assignments=False, **kwargs):
        assert return_assignments
        x = keras.ops.take(x, assignments, axis=0)
        if conditions is not None:
            conditions = keras.ops.take(conditions, assignments, axis=0)
        return z, x, conditions, assignments

    captured = {}
    weighted_mean = mean_flow_module.weighted_mean

    def capture_weights(elements, weights=None):
        captured["weights"] = weights
        return weighted_mean(elements, weights)

    monkeypatch.setattr(mean_flow_module, "optimal_transport", fake_optimal_transport)
    monkeypatch.setattr(mean_flow_module, "weighted_mean", capture_weights)

    network = make_mean_flow(random_samples, random_conditions, use_optimal_transport=True)
    sample_weight = keras.ops.arange(1, keras.ops.shape(random_samples)[0] + 1, dtype=random_samples.dtype)
    sample_weight = keras.ops.expand_dims(sample_weight, axis=-1)
    network.compute_metrics(random_samples, conditions=random_conditions, sample_weight=sample_weight)

    assert_allclose(captured["weights"], keras.ops.take(sample_weight, assignments, axis=0))


def test_time_schedule_at_call_time(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions, steps=4)
    samples = network(random_samples, conditions=random_conditions, inverse=True)
    samples_2 = network(random_samples, conditions=random_conditions, inverse=True, steps=2)
    assert max_abs_diff(samples, samples_2) > 1e-6

    time_schedule = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(2)).tolist()
    assert_allclose(
        network(random_samples, conditions=random_conditions, inverse=True, time_schedule=time_schedule), samples_2
    )

    time_schedule = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(2, rho=3.0, sigma_min=1e-2)).tolist()
    assert_allclose(
        network(random_samples, conditions=random_conditions, inverse=True, steps=2, rho=3.0, sigma_min=1e-2),
        network(random_samples, conditions=random_conditions, inverse=True, time_schedule=time_schedule),
    )


@pytest.mark.skip_on_mps
@pytest.mark.parametrize("steps", [1, 3])
def test_density_uses_exact_log_det(steps, random_samples, random_conditions):
    from bayesflow.utils import jacobian, log_abs_det

    network = make_mean_flow(random_samples, random_conditions)

    def f(x):
        return network(x, conditions=random_conditions, steps=steps)

    output, jac = jacobian(f, random_samples, return_output=True)
    _, log_density = network(random_samples, conditions=random_conditions, density=True, steps=steps)
    expected = network.base_distribution.log_prob(output) + log_abs_det(jac)
    assert_allclose(log_density, expected, rtol=1e-4, atol=1e-4)


def test_instantaneous_flow_adaptive(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions)
    batch_size = keras.ops.shape(random_samples)[0]

    log_prob = network.log_prob(random_samples, conditions=random_conditions, use_instantaneous_flow=True)
    assert keras.ops.shape(log_prob) == (batch_size,)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(log_prob)))

    samples = network.sample((batch_size,), conditions=random_conditions, use_instantaneous_flow=True)
    assert keras.ops.shape(samples) == keras.ops.shape(random_samples)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(samples)))


@pytest.mark.parametrize("method_at_call_time", [False, True])
def test_instantaneous_flow_forward_glass_falls_back_to_tsit5(method_at_call_time, random_samples, random_conditions):
    integrate_kwargs = None if method_at_call_time else {"method": "glass", "steps": 2}
    network = make_mean_flow(random_samples, random_conditions, integrate_kwargs=integrate_kwargs)
    kwargs = dict(conditions=random_conditions, use_instantaneous_flow=True, steps=2)
    if method_at_call_time:
        kwargs["method"] = "glass"

    output = network(random_samples, **kwargs)
    expected = network(random_samples, **(kwargs | {"method": "tsit5"}))

    assert_allclose(output, expected, atol=1e-6)


def test_serialize_deserialize(random_samples, random_conditions):
    network = make_mean_flow(random_samples, random_conditions)

    clone = build(deserialize(serialize(network)), random_samples, random_conditions)
    assert len(clone.velocity_blocks) == len(network.velocity_blocks)
    clone.set_weights(network.get_weights())

    assert_allclose(
        clone.velocity(random_samples, 0.3, conditions=random_conditions),
        network.velocity(random_samples, 0.3, conditions=random_conditions),
        atol=1e-6,
    )
    assert_allclose(
        clone(random_samples, conditions=random_conditions, inverse=True),
        network(random_samples, conditions=random_conditions, inverse=True),
        atol=1e-6,
    )


def test_save_load_output_unchanged(tmp_path, random_samples, random_conditions):
    """Loaded model produces the same output as the original."""
    network = make_mean_flow(random_samples, random_conditions)

    z = keras.random.normal(keras.ops.shape(random_samples))
    samples = network(z, conditions=random_conditions, inverse=True)
    velocity = network.velocity(z, 0.3, conditions=random_conditions)

    path = tmp_path / "mean_flow.keras"
    keras.saving.save_model(network, path)
    loaded = keras.saving.load_model(path)

    assert_allclose(samples, loaded(z, conditions=random_conditions, inverse=True), atol=1e-5, rtol=1e-5)
    assert_allclose(velocity, loaded.velocity(z, 0.3, conditions=random_conditions), atol=1e-5, rtol=1e-5)
