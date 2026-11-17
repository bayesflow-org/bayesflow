import numpy as np
import pytest

import keras


pm = pytest.importorskip("pymc")

pytestmark = pytest.mark.jax


@pytest.fixture()
def estimator():
    from bayesflow import BasicEstimator
    from bayesflow.networks import CouplingFlow

    feature_dim = 4
    condition_dim = 2
    estimator = BasicEstimator(
        inference_network=CouplingFlow(
            depth=1,
            transform="affine",
            subnet_kwargs={"widths": (8,)},
        ),
        standardize="all",
    )
    estimator.build(
        {
            "inference_variables": (2, feature_dim),
            "inference_conditions": (2, condition_dim),
        }
    )
    return estimator


def _build_ratio_estimator(observation_dim):
    from bayesflow import RatioEstimator
    from bayesflow.networks import MLP

    estimator = RatioEstimator(
        inference_network=MLP(widths=(8,)),
        K=1,
        standardize="all",
    )
    estimator.build(
        {
            "inference_variables": (2, 2),
            "inference_conditions": (2, observation_dim),
        }
    )
    return estimator


def _compile_logp_and_gradient(estimator, observed, *, exchangeable):
    from bayesflow.wrappers.pymc import NeuralDistribution

    neural_dist = NeuralDistribution(
        estimator,
        param_names=["theta_0", "theta_1"],
        exchangeable=exchangeable,
    )

    with pm.Model() as model:
        theta_0 = pm.Normal("theta_0", dtype=keras.config.floatx())
        theta_1 = pm.Normal("theta_1", dtype=keras.config.floatx())
        likelihood = neural_dist(
            "likelihood",
            theta_0=theta_0,
            theta_1=theta_1,
            observed=observed,
        )

    point = model.initial_point()
    logp = model.compile_logp(vars=[likelihood])(point)
    gradient = model.compile_dlogp()(point)
    return logp, gradient, neural_dist


@pytest.mark.parametrize("keep_batch_axis", [False, True])
def test_non_exchangeable_logp_accepts_single_observation(estimator, keep_batch_axis):
    observed = np.arange(4, dtype=keras.config.floatx()) / 10
    if keep_batch_axis:
        observed = observed[None, :]

    logp, gradient, _ = _compile_logp_and_gradient(estimator, observed, exchangeable=False)

    inference_variables = np.atleast_2d(observed)
    inference_conditions = np.zeros((1, 2), dtype=keras.config.floatx())
    expected = estimator.inference_network.log_prob(
        inference_variables,
        conditions=inference_conditions,
    )

    np.testing.assert_allclose(logp, np.asarray(expected).sum(), rtol=1e-5)
    assert np.isfinite(gradient).all()


def test_exchangeable_logp_sums_replicates(estimator):
    observed = np.arange(12, dtype=keras.config.floatx()).reshape(3, 4) / 10

    logp, gradient, _ = _compile_logp_and_gradient(estimator, observed, exchangeable=True)

    inference_conditions = np.zeros((3, 2), dtype=keras.config.floatx())
    expected = estimator.inference_network.log_prob(
        observed,
        conditions=inference_conditions,
    )

    np.testing.assert_allclose(logp, np.asarray(expected).sum(), rtol=1e-5)
    assert np.isfinite(gradient).all()


@pytest.mark.parametrize("observation_dim", [1, 2])
def test_exchangeable_ratio_logp_sums_scalar_and_vector_observations(observation_dim):
    estimator = _build_ratio_estimator(observation_dim)
    shape = (3,) if observation_dim == 1 else (3, observation_dim)
    observed = np.arange(np.prod(shape), dtype=keras.config.floatx()).reshape(shape) / 10

    logp, gradient, neural_dist = _compile_logp_and_gradient(estimator, observed, exchangeable=True)

    expected = sum(neural_dist.backend.log_prob(x, 0.0, 0.0) for x in observed)
    np.testing.assert_allclose(logp, np.asarray(expected), rtol=1e-5)
    assert np.isfinite(gradient).all()


def test_exchangeable_ratio_logp_accepts_per_trial_parameter():
    from bayesflow.wrappers.pymc import NeuralDistribution

    estimator = _build_ratio_estimator(observation_dim=1)
    observed = np.arange(3, dtype=keras.config.floatx()) / 10
    condition = np.array([0.0, 1.0, 1.0], dtype=keras.config.floatx())
    neural_dist = NeuralDistribution(
        estimator,
        param_names=["mu", "sigma"],
        exchangeable=True,
    )

    with pm.Model() as model:
        beta_0 = pm.Normal("beta_0", dtype=keras.config.floatx())
        beta_1 = pm.Normal("beta_1", dtype=keras.config.floatx())
        sigma = pm.Normal("sigma", dtype=keras.config.floatx())
        likelihood = neural_dist(
            "likelihood",
            mu=beta_0 + beta_1 * condition,
            sigma=sigma,
            observed=observed,
        )

    point = model.initial_point()
    logp = model.compile_logp(vars=[likelihood])(point)
    gradient = model.compile_dlogp()(point)
    expected = sum(neural_dist.backend.log_prob(x, 0.0, 0.0) for x in observed)

    np.testing.assert_allclose(logp, np.asarray(expected), rtol=1e-5)
    assert np.isfinite(gradient).all()
