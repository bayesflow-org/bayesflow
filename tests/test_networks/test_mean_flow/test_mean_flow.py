import keras
import numpy as np
import pytest

from bayesflow.experimental import MeanFlow
from bayesflow.masking import InferenceMasks
from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_allclose, skip_torch_linalg_on_mps


def _build(network, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    network.build(xz_shape, conditions_shape=cond_shape)
    return network


def _mean_flow(random_samples, random_conditions, **kwargs):
    kwargs.setdefault("subnet_kwargs", dict(widths=(8, 8)))
    return _build(MeanFlow(**kwargs), random_samples, random_conditions)


def _max_abs_diff(x1, x2):
    return np.max(np.abs(keras.ops.convert_to_numpy(x1) - keras.ops.convert_to_numpy(x2)))


# ---- Time schedule ---------------------------------------------------------


@pytest.mark.parametrize("steps", [1, 4])
def test_make_time_schedule(steps):
    times = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(steps))
    assert times.shape == (steps + 1,)
    np.testing.assert_allclose(times[[0, -1]], [1.0, 0.0], atol=1e-6)
    assert np.all(np.diff(times) < 0.0)


# ---- Maps ------------------------------------------------------------------


def test_flow_map_is_average_velocity_step(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions)
    u = network._average_velocity(random_samples, 1.0, 0.25, random_conditions, False, InferenceMasks())
    mapped = network._flow_map(random_samples, 1.0, 0.25, random_conditions, False, InferenceMasks())
    assert_allclose(mapped, random_samples + (0.25 - 1.0) * u, atol=1e-6)


def test_velocity_without_head_is_boundary_average_velocity(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions, velocity_head_layers=0)
    v = network.velocity(random_samples, 0.4, conditions=random_conditions)
    u = network._average_velocity(random_samples, 0.4, 0.4, random_conditions, False, InferenceMasks())
    np.testing.assert_array_equal(keras.ops.convert_to_numpy(v), keras.ops.convert_to_numpy(u))


# ---- Velocity head ---------------------------------------------------------


def test_velocity_head(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions, velocity_head_layers=1)
    leading_block, trailing_block = network.subnet.blocks
    (head_block,) = network.velocity_blocks

    # the head owns a clone of the trailing block
    assert type(head_block) is type(trailing_block)
    assert head_block.get_config() | {"name": None} == trailing_block.get_config() | {"name": None}
    head_weights = {id(w) for w in head_block.weights}
    assert head_weights and head_weights.isdisjoint(id(w) for w in network.subnet.weights)

    v = network.velocity(random_samples, 0.4, conditions=random_conditions)
    u = network._average_velocity(random_samples, 0.4, 0.4, random_conditions, False, InferenceMasks())
    assert _max_abs_diff(v, u) > 1e-3

    # and shares the leading block with the subnet
    for weight in leading_block.weights:
        weight.assign(keras.random.normal(weight.shape))
    assert _max_abs_diff(network.velocity(random_samples, 0.4, conditions=random_conditions), v) > 1e-3


def test_velocity_head_after_shared_blocks_of_other_width(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions, subnet_kwargs=dict(widths=(8, 16)), velocity_head_layers=1)
    v = network.velocity(random_samples, 0.4, conditions=random_conditions)
    assert keras.ops.shape(v) == keras.ops.shape(random_samples)


def test_velocity_head_raises():
    with pytest.raises(ValueError, match="embed"):
        MeanFlow(subnet="mlp")
    with pytest.raises(ValueError, match="velocity_head_layers"):
        MeanFlow(subnet_kwargs=dict(widths=(8, 8)), velocity_head_layers=3)


# ---- compute_metrics (training path of each variant) ------------------------


def test_compute_metrics(mean_flow, random_samples, random_conditions):
    if mean_flow.use_optimal_transport and random_conditions is not None:
        skip_torch_linalg_on_mps()

    _build(mean_flow, random_samples, random_conditions)
    metrics = mean_flow.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_with_sample_weight(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions)
    sample_weight = keras.random.uniform(keras.ops.shape(random_samples)[:1])
    metrics = network.compute_metrics(random_samples, conditions=random_conditions, sample_weight=sample_weight)
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


# ---- Time schedule at sampling and density time ----------------------------


def test_time_schedule_at_call_time(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions, steps=4)
    samples = network(random_samples, conditions=random_conditions, inverse=True)
    samples_2 = network(random_samples, conditions=random_conditions, inverse=True, steps=2)
    assert _max_abs_diff(samples, samples_2) > 1e-6

    time_schedule = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(2)).tolist()
    assert_allclose(
        network(random_samples, conditions=random_conditions, inverse=True, time_schedule=time_schedule), samples_2
    )

    time_schedule = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(2, rho=3.0, sigma_min=1e-2)).tolist()
    assert_allclose(
        network(random_samples, conditions=random_conditions, inverse=True, steps=2, rho=3.0, sigma_min=1e-2),
        network(random_samples, conditions=random_conditions, inverse=True, time_schedule=time_schedule),
    )

    with pytest.raises(ValueError, match="two times"):
        network(random_samples, conditions=random_conditions, inverse=True, time_schedule=[1.0])


@pytest.mark.skip_on_mps
@pytest.mark.parametrize("steps", [1, 3])
def test_density_uses_exact_log_det(steps, random_samples, random_conditions):
    from bayesflow.utils import jacobian, log_abs_det

    network = _mean_flow(random_samples, random_conditions)

    def f(x):
        return network(x, conditions=random_conditions, steps=steps)

    output, jac = jacobian(f, random_samples, return_output=True)
    _, log_density = network(random_samples, conditions=random_conditions, density=True, steps=steps)
    expected = network.base_distribution.log_prob(output) + log_abs_det(jac)
    assert_allclose(log_density, expected, rtol=1e-4, atol=1e-4)


# ---- Density head ----------------------------------------------------------


def test_density_head_log_prob(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions, density_head=True)

    # the forward pass runs from data at t = 0 to noise at t = 1
    times = keras.ops.convert_to_numpy(MeanFlow.make_time_schedule(2))[::-1]
    x, log_det = random_samples, 0.0
    for source, target in zip(times[:-1], times[1:]):
        u, divergence = network._average_velocity_and_divergence(
            x, source, target, random_conditions, False, InferenceMasks()
        )
        x = x + (target - source) * u
        log_det += (target - source) * keras.ops.sum(divergence, axis=-1)

    z, log_density = network(random_samples, conditions=random_conditions, density=True, steps=2)
    assert_allclose(z, x, atol=1e-5)
    assert_allclose(log_density, network.base_distribution.log_prob(x) + log_det, atol=1e-4)


@pytest.mark.parametrize("subnet", ["time_mlp", "diffusion_transformer"])
def test_density_head_shares_subnet_pass(subnet, random_samples, random_conditions):
    if subnet == "time_mlp":
        # a final width equal to the target dimension, where the average velocity has no projector
        network = _mean_flow(random_samples, random_conditions, subnet_kwargs=dict(widths=(8, 2)), density_head=True)
    else:
        network = _transformer_mean_flow(random_samples, random_conditions, density_head=True)

    u, divergence = network._average_velocity_and_divergence(
        random_samples, 0.8, 0.3, random_conditions, False, InferenceMasks()
    )
    expected = network._average_velocity(random_samples, 0.8, 0.3, random_conditions, False, InferenceMasks())
    assert_allclose(u, expected, atol=1e-6)
    assert _max_abs_diff(u, divergence) > 1e-3


# ---- Instantaneous flow ----------------------------------------------------


def test_instantaneous_flow_adaptive(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions)
    batch_size = keras.ops.shape(random_samples)[0]

    log_prob = network.log_prob(random_samples, conditions=random_conditions, use_instantaneous_flow=True)
    assert keras.ops.shape(log_prob) == (batch_size,)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(log_prob)))

    samples = network.sample((batch_size,), conditions=random_conditions, use_instantaneous_flow=True)
    assert keras.ops.shape(samples) == keras.ops.shape(random_samples)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(samples)))


# ---- Serialization ---------------------------------------------------------


def test_serialize_deserialize(random_samples, random_conditions):
    network = _mean_flow(random_samples, random_conditions)

    clone = _build(deserialize(serialize(network)), random_samples, random_conditions)
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
    network = _mean_flow(random_samples, random_conditions)

    z = keras.random.normal(keras.ops.shape(random_samples))
    samples = network(z, conditions=random_conditions, inverse=True)
    velocity = network.velocity(z, 0.3, conditions=random_conditions)

    path = tmp_path / "mean_flow.keras"
    keras.saving.save_model(network, path)
    loaded = keras.saving.load_model(path)

    assert_allclose(samples, loaded(z, conditions=random_conditions, inverse=True), atol=1e-5, rtol=1e-5)
    assert_allclose(velocity, loaded.velocity(z, 0.3, conditions=random_conditions), atol=1e-5, rtol=1e-5)


# ---- Diffusion transformer subnet ------------------------------------------


def _transformer_mean_flow(random_samples, random_conditions, **kwargs):
    network = _mean_flow(
        random_samples,
        random_conditions,
        subnet="diffusion_transformer",
        subnet_kwargs=dict(widths=(8, 8), num_heads=2),
        **kwargs,
    )

    # the readouts are zero-initialized, so an untrained network would predict zero
    readouts = (network.subnet, network.velocity_readout, network.divergence_readout)
    for readout in filter(None, readouts):
        for weight in readout.token_out.weights + readout.final_ada_ln.weights:
            weight.assign(keras.random.normal(weight.shape, stddev=0.3))
    return network


def test_transformer_velocity_readout(random_samples, random_conditions):
    from bayesflow.networks.helpers import DiffusionTransformerReadout

    network = _transformer_mean_flow(random_samples, random_conditions)
    assert type(network.velocity_readout) is DiffusionTransformerReadout

    readout_weights = {id(w) for w in network.velocity_readout.weights}
    assert readout_weights and readout_weights.isdisjoint(id(w) for w in network.subnet.weights)


def test_transformer_reads_target_time(random_samples, random_conditions):
    network = _transformer_mean_flow(random_samples, random_conditions)
    u = network._average_velocity(random_samples, 0.8, 0.8, random_conditions, False, InferenceMasks())
    u_moved = network._average_velocity(random_samples, 0.8, 0.2, random_conditions, False, InferenceMasks())

    assert np.std(keras.ops.convert_to_numpy(u)) > 1e-3
    assert _max_abs_diff(u, u_moved) > 1e-4


def test_transformer_compute_metrics(random_samples, random_conditions):
    if keras.backend.backend() == "torch":
        import torch

        if not torch.cuda.is_available():
            pytest.skip("PyTorch CPU attention does not support the forward-mode AD used by mean flows.")

    network = _transformer_mean_flow(random_samples, random_conditions)
    loss = keras.ops.convert_to_numpy(network.compute_metrics(random_samples, conditions=random_conditions)["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"
