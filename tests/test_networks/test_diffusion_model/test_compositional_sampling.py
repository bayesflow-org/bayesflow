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
    simple_compositional_diffusion_model.build(state_shape, (state_shape[0],) + conditions_shape[2:])
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
        mini_batch_size=2,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )

    # Test inverse sampling without prior score
    result = simple_compositional_diffusion_model._inverse_compositional(
        z=compositional_state,
        conditions=compositional_conditions,
        compute_prior_score=None,
        density=False,
        training=False,
        method="euler_maruyama",
        steps=10,
        start_time=1.0,
        stop_time=0.0,
        mini_batch_size=2,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )

    # Test inverse sampling with jacobian approximation (expensive)
    result = simple_compositional_diffusion_model._inverse_compositional(
        z=compositional_state,
        conditions=compositional_conditions,
        compute_prior_score=None,
        density=False,
        training=False,
        method="euler_maruyama",
        steps=5,
        start_time=1.0,
        stop_time=0.0,
        mini_batch_size=2,
        use_jac=True,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


def test_ancestral_sampling():
    """
    Hierarchical scenario:
      global:  mu ~ N(0, 1)                      (shared across subjects)
      local:   beta_i ~ N(mu, 0.1),  x_i = beta_i + noise

    The local-level workflow approximates p(beta | mu, x).
    Global posterior samples are passed as ancestral_conditions; per-subject
    observations as conditions. The function should expand both and return
    samples of shape (n_test, n_subjects, num_samples, 1).
    """
    from bayesflow.networks import ConsistencyModel
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import Simulator

    class LocalSimulator(Simulator):
        def sample(self, n, local_n=1, **kwargs):
            mu = np.random.randn(n, 1)
            beta = mu + 0.1 * np.random.randn(n, local_n)
            x = beta + 0.1 * np.random.randn(n, local_n)
            return {"mu": mu, "beta": beta, "x": x}

    workflow = BasicWorkflow(
        inference_network=ConsistencyModel(subnet_kwargs=dict(widths=(8, 8)), total_steps=5 * 2),
        inference_variables=["beta"],
        inference_conditions=["mu", "x"],
        simulator=LocalSimulator(),
    )
    workflow.fit_online(epochs=5, batch_size=4, num_batches_per_epoch=2, verbose=0)

    n_test = 3
    local_n = 4
    num_samples = 2

    # Global posterior samples from an upper-level approximator (pre-computed)
    ancestral_conditions = {"mu": np.random.randn(n_test, num_samples, 1)}
    # Per-subject local observations
    conditions = {"x": np.random.randn(n_test, local_n, 1)}

    samples = workflow.ancestral_sample(
        conditions=conditions,
        ancestral_conditions=ancestral_conditions,
    )

    assert "beta" in samples
    assert samples["beta"].shape == (n_test, local_n, num_samples, 1)


# ---- Guidance (slower) ----------------------------------------


def test_compositional_masking():
    from bayesflow.networks import CompositionalDiffusionModel
    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    num_samples = 3
    batch_size = 2
    num_batches_per_epoch = 2
    epochs = 5
    workflow = BasicWorkflow(
        inference_network=CompositionalDiffusionModel(
            subnet_kwargs=dict(widths=(8, 8)),
            drop_target_prob=0.5,
        ),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=epochs, batch_size=batch_size, num_batches_per_epoch=num_batches_per_epoch)
    test_params = workflow.simulate(5)["parameters"]
    test_conditions = {
        "observables": np.array(
            [
                (TwoMoons().observation_model(t), TwoMoons().observation_model(t), TwoMoons().observation_model(t))
                for t in test_params
            ]
        )
    }
    test_conditions.update({"parameters": test_params})

    def prior_score_fn(theta):
        # uniform prior (should be transformed to unbounded prior for a real application)
        return {"parameters": keras.ops.zeros(keras.ops.shape(theta["parameters"]))}

    samples = workflow.compositional_sample(
        num_samples=num_samples, conditions=test_conditions, compute_prior_score=prior_score_fn
    )["parameters"]

    test_conditions_adapted = workflow.adapter(test_conditions)
    target_mask = keras.ops.concatenate(
        (
            keras.ops.ones(1),  # param 1 is inferred
            keras.ops.zeros(1),  # param 2 is fixed
        )
    )
    targets_fixed = test_conditions_adapted["inference_variables"][0]  # one set of parameters
    if "inference_variables" in workflow.approximator.standardize_layers:
        targets_fixed = workflow.approximator.standardize_layers["inference_variables"](targets_fixed, forward=True)

    fixed_samples = workflow.compositional_sample(
        conditions=test_conditions,
        num_samples=num_samples,
        compute_prior_score=prior_score_fn,
        targets_fixed=targets_fixed,
        target_mask=target_mask,
    )["parameters"]
    assert samples.shape == fixed_samples.shape
    assert (np.abs(fixed_samples[..., 1] - test_conditions["parameters"][0, 1]) < 1e-6).all()
    assert (np.abs(fixed_samples[..., 0] - test_conditions["parameters"][0, 0]) > 0.1).any()  # should vary


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
