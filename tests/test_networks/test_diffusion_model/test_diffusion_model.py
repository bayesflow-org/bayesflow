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


def test_mask_aware_subnet_receives_target_inference_mask(random_samples, random_conditions):
    from bayesflow.networks import DiffusionModel

    class MaskAwareSubnet(keras.Layer):
        def __init__(self):
            super().__init__()
            self.last_target_inference_mask = None

        def call(self, inputs, training=None, target_inference_mask=None):
            self.last_target_inference_mask = target_inference_mask
            return inputs[0]

        def compute_output_shape(self, input_shape):
            return input_shape[0]

    subnet = MaskAwareSubnet()
    model = DiffusionModel(subnet=subnet, drop_target_prob=0.5)
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    model.build(xz_shape, conditions_shape=cond_shape)

    model.compute_metrics(random_samples, conditions=random_conditions)
    assert subnet.last_target_inference_mask is not None
    assert keras.ops.shape(subnet.last_target_inference_mask) == keras.ops.shape(random_samples)

    target_inference_mask = keras.ops.ones_like(random_samples)
    model.velocity(
        random_samples,
        time=0.5,
        stochastic_solver=False,
        conditions=random_conditions,
        target_inference_mask=target_inference_mask,
    )
    assert np.allclose(
        keras.ops.convert_to_numpy(subnet.last_target_inference_mask), keras.ops.convert_to_numpy(target_inference_mask)
    )


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

    workflow.approximator.inference_network.guidance_function = guidance_function
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


def test_tdiffusion_transformer_condition_mask_forwarded():
    """A per-condition condition_mask is accepted and the output keeps the target shape."""
    from bayesflow.networks import DiffusionModel

    dm = DiffusionModel(subnet="diffusion_transformer", subnet_kwargs=dict(widths=(8, 8)))
    x = keras.random.normal((4, 5))
    cond = keras.random.normal((4, 3))
    dm.build(keras.ops.shape(x), keras.ops.shape(cond))

    condition_mask = keras.ops.convert_to_tensor(
        np.array([[1, 1, 0]] * 4, dtype="float32")  # third condition missing
    )
    metrics = dm.compute_metrics(x, conditions=cond, stage="training", condition_mask=condition_mask)
    assert np.isfinite(keras.ops.convert_to_numpy(metrics["loss"]))


def test_drop_missing_prob_training_and_serialization():
    """Missingness training runs and the flag round-trips through serialization."""
    from bayesflow.networks import DiffusionModel

    dm = DiffusionModel(
        subnet="diffusion_transformer",
        subnet_kwargs=dict(widths=(8, 8)),
        drop_target_prob=0.5,
        drop_missing_prob=0.3,
    )
    x = keras.random.normal((4, 5))
    cond = keras.random.normal((4, 3))
    dm.build(keras.ops.shape(x), keras.ops.shape(cond))

    metrics = dm.compute_metrics(x, conditions=cond, stage="training")
    assert np.isfinite(keras.ops.convert_to_numpy(metrics["loss"]))

    restored = deserialize(serialize(dm))
    assert restored.drop_missing_prob == 0.3
    # Missingness must not fire outside training.
    metrics_val = dm.compute_metrics(x, conditions=cond, stage="validation")
    assert np.isfinite(keras.ops.convert_to_numpy(metrics_val["loss"]))
