import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_allclose, assert_layers_equal


# ---- Noise schedule tests --------------------------------------------------


def test_serialize_deserialize_noise_schedule(noise_schedule):
    serialized = serialize(noise_schedule)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert serialized == reserialized
    t = 0.251
    x = 0.5
    training = True
    assert noise_schedule.log_snr(t, training=training) == deserialized.log_snr(t, training=training)
    assert noise_schedule.time_from_log_snr(t, training=training) == deserialized.time_from_log_snr(
        t, training=training
    )
    assert noise_schedule.derivative_log_snr(t, training=False) == deserialized.derivative_log_snr(t, training=False)
    assert noise_schedule.get_drift(t, x, training=False) == deserialized.get_drift(t, x, training=False)
    assert noise_schedule.get_alpha_sigma(t) == deserialized.get_alpha_sigma(t)
    assert noise_schedule.weights_for_snr(t) == deserialized.weights_for_snr(t)


def test_validate_noise_schedule(noise_schedule):
    noise_schedule.validate()


def test_cosine_noise_schedule_inverse(cosine_noise_schedule):
    t = keras.ops.linspace(0.0, 1.0, 11)
    log_snr = cosine_noise_schedule.log_snr(t, training=False)
    reconstructed_t = cosine_noise_schedule.time_from_log_snr(log_snr, training=False)

    expected_endpoints = keras.ops.convert_to_tensor(
        [cosine_noise_schedule.log_snr_max, cosine_noise_schedule.log_snr_min]
    )
    actual_endpoints = keras.ops.stack([log_snr[0], log_snr[-1]])
    assert_allclose(actual_endpoints, expected_endpoints, atol=1e-5)
    assert_allclose(reconstructed_t, t)


@pytest.mark.parametrize("variance_type", ["preserving", "exploding"])
def test_ode_sampling_with_exact_score(variance_type):
    """With the exact score of N(0, s^2), the probability flow ODE recovers N(0, s^2)."""
    from bayesflow.networks import DiffusionModel
    from bayesflow.networks.inference.diffusion.schedules import EDMNoiseSchedule

    s = 0.5
    model = DiffusionModel(
        noise_schedule=EDMNoiseSchedule(variance_type=variance_type),
        subnet_kwargs=dict(widths=(8,)),
        integrate_kwargs=dict(method="tsit5", steps=50),
    )
    model.build((2048, 1))

    def exact_score(xz, *, log_snr_t, **kwargs):
        alpha, sigma = model.noise_schedule.get_alpha_sigma(log_snr_t)
        return -xz / (alpha**2 * s**2 + sigma**2)

    model._score = exact_score
    samples = keras.ops.convert_to_numpy(model.sample(2048, seed=0))
    assert np.isclose(np.std(samples), s, rtol=0.1)


# ---- Configuration ---------------------------------------------------------


def test_build_with_custom_integrate_kwargs():
    from bayesflow.networks import DiffusionModel

    model = DiffusionModel(
        subnet_kwargs=dict(widths=(8, 8)),
        integrate_kwargs=dict(method="euler", steps=10),
    )
    model.build((2, 3), conditions_shape=(2, 3))
    assert model.built
    assert model.integrate_kwargs["method"] == "euler"
    assert model.integrate_kwargs["steps"] == 10


# ---- Prediction type / noise schedule variants ------------------------------
# The generic interface contract (shapes, batch sizes, save/load, ...) is
# covered by test_networks/test_inference_networks.py for the default model.
# The tests below run the model-specific configuration variants through the
# training and (fixed-step, low-accuracy) sampling/density code paths.


def test_compute_metrics(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = diffusion_model.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


@pytest.mark.parametrize("loss_type", ["noise", "velocity", "F"])
def test_compute_metrics_loss_types(loss_type, random_samples, random_conditions):
    from bayesflow.networks import DiffusionModel

    model = DiffusionModel(subnet_kwargs=dict(widths=(8, 8)), loss_type=loss_type)
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    model.build(xz_shape, conditions_shape=cond_shape)

    metrics = model.compute_metrics(random_samples, conditions=random_conditions)
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_sample_and_density(diffusion_model, random_samples, random_conditions):
    """Each prediction type supports sampling and density evaluation in both directions.

    Accuracy is checked in test_inference_networks.py::test_density_numerically;
    here a few fixed solver steps suffice to exercise the velocity/score conversions.
    """
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)
    diffusion_model.integrate_kwargs.update({"method": "rk45", "steps": 8})

    z = keras.random.normal(xz_shape)
    x, log_density = diffusion_model(z, conditions=random_conditions, inverse=True, density=True)
    assert keras.ops.shape(x) == xz_shape
    assert keras.ops.shape(log_density) == (xz_shape[0],)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(x)))
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(log_density)))

    z, log_density = diffusion_model(random_samples, conditions=random_conditions, density=True)
    assert keras.ops.shape(z) == xz_shape
    assert keras.ops.shape(log_density) == (xz_shape[0],)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(z)))
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(log_density)))


def test_adaptive_solver_sample_and_density():
    """The instance defaults are adaptive solvers (stochastic sampling, adaptive ODE for
    density). The adaptive density must match a high-accuracy fixed-step reference, which
    in turn is verified against a numerical jacobian in
    test_inference_networks.py::test_density_numerically."""
    from bayesflow.networks import DiffusionModel

    model = DiffusionModel(subnet_kwargs=dict(widths=(8, 8), fourier_scale=1.0))
    conditions = keras.random.normal((2, 3))
    model.build((2, 3), conditions_shape=(2, 3))

    # sampling with the default (stochastic, adaptive) solver
    z = keras.random.normal((2, 3))
    x = model(z, conditions=conditions, inverse=True)
    assert keras.ops.shape(x) == (2, 3)
    assert np.all(np.isfinite(keras.ops.convert_to_numpy(x)))

    # density with the adaptive ODE solver, in both directions
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


def test_serialize_deserialize(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    serialized = serialize(diffusion_model)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert keras.tree.lists_to_tuples(serialized) == keras.tree.lists_to_tuples(reserialized)


def test_save_and_load(tmp_path, diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    path = tmp_path / "diffusion.keras"
    keras.saving.save_model(diffusion_model, path)
    loaded = keras.saving.load_model(path)

    assert_layers_equal(diffusion_model, loaded)


# ---- Guidance (slow, trains a model) ----------------------------------------


@pytest.mark.slow
def test_diffusion_guidance(simple_diffusion_model):
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    workflow = BasicWorkflow(
        inference_network=simple_diffusion_model,
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=2, batch_size=2, num_batches_per_epoch=2, verbose=0)
    test_conditions = workflow.simulate(5)
    samples = workflow.sample(num_samples=2, conditions=test_conditions)["parameters"]

    def constraint(params):
        # params are automatically unstandardized before the constraint is called
        a1 = params[..., 0]
        return a1

    samples_guided = workflow.sample(
        num_samples=2, conditions=test_conditions, guidance_kwargs=dict(constraints=constraint)
    )["parameters"]
    assert samples_guided.shape == samples.shape
    assert (samples_guided[..., 0] < 0).all()

    def guidance_function(x_pred, time, score, **guidance_kwargs):
        # params are not automatically unstandardized before the guidance is called
        unstandardize = guidance_kwargs.get("unstandardize", lambda x: x)
        x_pred = unstandardize(x_pred)
        return x_pred * 0

    workflow.estimator.inference_network.guidance_function = guidance_function
    samples_guided_func = workflow.sample(
        num_samples=2,
        conditions=test_conditions,
    )["parameters"]
    assert samples_guided_func.shape == samples.shape


# ---- Joint condition tokenization -----------------------------------------


def test_diffusion_transformer_conditions_as_tokens_training_step():
    """Conditions are tokenized per-dimension; the diffusion output still matches the
    target shape and a training step produces a finite loss."""
    from bayesflow.networks import DiffusionModel

    dm = DiffusionModel(subnet="diffusion_transformer", subnet_kwargs=dict(widths=(8, 8)))
    x = keras.random.normal((4, 5))
    cond = keras.random.normal((4, 3))
    dm.build(keras.ops.shape(x), keras.ops.shape(cond))

    metrics = dm.compute_metrics(x, conditions=cond, stage="training")
    assert np.isfinite(keras.ops.convert_to_numpy(metrics["loss"]))

    restored = deserialize(serialize(dm))
    metrics_r = restored.compute_metrics(x, conditions=cond, stage="training")
    assert np.isfinite(keras.ops.convert_to_numpy(metrics_r["loss"]))
