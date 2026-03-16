import keras
import pytest
import numpy as np


def test_compositional_score_shape(
    simple_compositional_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test that compositional score returns correct shapes."""
    # Build the model
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_compositional_diffusion_model.build(state_shape, conditions_shape)
    simple_compositional_diffusion_model.compositional_bridge_d0 = 1
    simple_compositional_diffusion_model.compositional_bridge_d1 = 0.1

    time = 0.5

    score = simple_compositional_diffusion_model.compositional_score(
        xz=compositional_state,
        time=time,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        training=False,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(score)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


def test_compositional_score_no_conditions_raises_error(
    simple_compositional_diffusion_model, compositional_state, mock_prior_score
):
    """Test that compositional score raises error when conditions is None."""
    simple_compositional_diffusion_model.build(keras.ops.shape(compositional_state), None)

    with pytest.raises(ValueError, match="Conditions are required for compositional sampling"):
        simple_compositional_diffusion_model.compositional_score(
            xz=compositional_state, time=0.5, conditions=None, compute_prior_score=mock_prior_score, training=False
        )


def test_inverse_compositional_basic(
    simple_compositional_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test basic compositional inverse sampling."""
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_compositional_diffusion_model.build(state_shape, conditions_shape)

    # Test inverse sampling
    result = simple_compositional_diffusion_model._inverse_compositional(
        z=compositional_state,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        density=False,
        training=False,
        method="euler_maruyama",
        steps=5,
        start_time=1.0,
        stop_time=0.0,
        mini_batch_size=1,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


# ---- Guidance (slow, trains a model) ----------------------------------------


@pytest.mark.slow
def test_diffusion_compositional_guidance():
    from bayesflow.networks import CompositionalDiffusionModel
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    workflow = BasicWorkflow(
        inference_network=CompositionalDiffusionModel(subnet_kwargs=dict(widths=(8, 8))),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=2, batch_size=2, num_batches_per_epoch=2, verbose=0)
    test_params = workflow.simulate(5)["parameters"]
    test_conditions = {
        "observables": np.array(
            [
                (TwoMoons().observation_model(t), TwoMoons().observation_model(t), TwoMoons().observation_model(t))
                for t in test_params
            ]
        )
    }

    def prior_score_fn(theta):
        # uniform prior (should be transformed to unbounded prior for a real application)
        return {"parameters": keras.ops.zeros(keras.ops.shape(theta["parameters"]))}

    samples = workflow.compositional_sample(
        num_samples=2, conditions=test_conditions, compute_prior_score=prior_score_fn
    )["parameters"]

    def constraint(z):
        params = workflow.approximator.standardize_layers["inference_variables"](z, forward=False)
        a1 = params[..., 0]
        return a1

    samples_guided = workflow.compositional_sample(
        num_samples=2,
        conditions=test_conditions,
        compute_prior_score=prior_score_fn,
        guidance_constraints=dict(constraints=constraint),
    )["parameters"]
    assert samples_guided.shape == samples.shape
    assert (samples_guided[..., 0] < 0).all()

    def guidance_function(x, time):
        return x * 0

    samples_guided_func = workflow.compositional_sample(
        num_samples=2,
        conditions=test_conditions,
        compute_prior_score=prior_score_fn,
        guidance_function=guidance_function,
    )["parameters"]
    assert samples_guided_func.shape == samples.shape
