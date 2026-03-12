import keras
import numpy as np
import pytest

from bayesflow.utils.serialization import serialize, deserialize
from tests.utils import assert_layers_equal


# ---- Noise schedule tests --------------------------------------------------


def test_serialize_deserialize_noise_schedule(noise_schedule):
    serialized = serialize(noise_schedule)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert serialized == reserialized
    t = 0.251
    x = 0.5
    training = True
    assert noise_schedule.get_log_snr(t, training=training) == deserialized.get_log_snr(t, training=training)
    assert noise_schedule.get_t_from_log_snr(t, training=training) == deserialized.get_t_from_log_snr(
        t, training=training
    )
    assert noise_schedule.derivative_log_snr(t, training=False) == deserialized.derivative_log_snr(t, training=False)
    assert noise_schedule.get_drift(t, x, training=False) == deserialized.get_drift(t, x, training=False)
    assert noise_schedule.get_alpha_sigma(t) == deserialized.get_alpha_sigma(t)
    assert noise_schedule.get_weights_for_snr(t) == deserialized.get_weights_for_snr(t)


def test_validate_noise_schedule(noise_schedule):
    noise_schedule.validate()


# ---- Build -----------------------------------------------------------------


def test_build(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None

    assert not diffusion_model.built
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)
    assert diffusion_model.built
    assert diffusion_model.variables


def test_build_with_custom_integrate_kwargs(random_samples, random_conditions):
    from bayesflow.networks import DiffusionModel

    model = DiffusionModel(
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


def test_inverse_output_shape(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    out = diffusion_model(z, conditions=random_conditions, inverse=True)
    assert keras.ops.shape(out) == keras.ops.shape(random_samples)


def test_inverse_density_output_shape(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z = keras.random.normal(keras.ops.shape(random_samples))
    x, log_density = diffusion_model(z, conditions=random_conditions, inverse=True, density=True)
    assert keras.ops.shape(x) == keras.ops.shape(random_samples)
    assert keras.ops.shape(log_density) == (keras.ops.shape(random_samples)[0],)


def test_forward_density_output_shape(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    z, log_density = diffusion_model(random_samples, conditions=random_conditions, density=True)
    assert keras.ops.shape(z) == keras.ops.shape(random_samples)
    assert keras.ops.shape(log_density) == (keras.ops.shape(random_samples)[0],)


# ---- Variable batch size ---------------------------------------------------


def test_variable_batch_size(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    for bs in [1, 4, 7]:
        z = keras.random.normal((bs,) + keras.ops.shape(random_samples)[1:])
        cond = (
            None if random_conditions is None else keras.random.normal((bs,) + keras.ops.shape(random_conditions)[1:])
        )
        out = diffusion_model(z, conditions=cond, inverse=True)
        assert keras.ops.shape(out)[0] == bs


# ---- Serialization ---------------------------------------------------------


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


# ---- compute_metrics -------------------------------------------------------


def test_compute_metrics(diffusion_model, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model.build(xz_shape, conditions_shape=cond_shape)

    metrics = diffusion_model.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


def test_compute_metrics_with_masking(diffusion_model_with_masking, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    diffusion_model_with_masking.build(xz_shape, conditions_shape=cond_shape)

    metrics = diffusion_model_with_masking.compute_metrics(random_samples, conditions=random_conditions)
    assert "loss" in metrics
    loss = keras.ops.convert_to_numpy(metrics["loss"])
    assert np.isfinite(loss), f"Loss is not finite: {loss}"


# ---- Guidance (slow, trains a model) ----------------------------------------


@pytest.mark.slow
def test_diffusion_guidance():
    from bayesflow.networks import DiffusionModel
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    workflow = BasicWorkflow(
        inference_network=DiffusionModel(subnet_kwargs=dict(widths=(8, 8))),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=2, batch_size=2, num_batches_per_epoch=2, verbose=0)
    test_conditions = workflow.simulate(5)
    samples = workflow.sample(num_samples=2, conditions=test_conditions)["parameters"]

    def constraint(z):
        params = workflow.approximator.standardize_layers["inference_variables"](z, forward=False)
        a1 = params[..., 0]
        return a1

    samples_guided = workflow.sample(
        num_samples=2, conditions=test_conditions, guidance_constraints=dict(constraints=constraint)
    )["parameters"]
    assert samples_guided.shape == samples.shape
    assert (samples_guided[..., 0] < 0).all()

    def guidance_function(x, time):
        return x * 0

    samples_guided_func = workflow.sample(
        num_samples=2, conditions=test_conditions, guidance_function=guidance_function
    )["parameters"]
    assert samples_guided_func.shape == samples.shape
