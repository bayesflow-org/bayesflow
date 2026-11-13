import warnings

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

import bayesflow as bf

arviz_stats = pytest.importorskip("arviz_stats")


# conjugate model: theta ~ N(0, prior_scale^2), x_i ~ N(theta, 1), i = 1..num_obs
PRIOR_SCALE = 2.0
NUM_OBS = 10


def analytic_posterior(x):
    """Posterior mean and scale per data set, x has shape (num_datasets, num_obs)."""
    precision = 1 / PRIOR_SCALE**2 + x.shape[-1]
    return x.sum(axis=-1) / precision, np.sqrt(1 / precision)


def log_prior(samples):
    return norm.logpdf(samples["theta"][..., 0], 0.0, PRIOR_SCALE)


def log_likelihood(samples, conditions):
    theta = samples["theta"]  # (num_datasets, num_samples, 1)
    x = conditions["x"][:, None, :]  # (num_datasets, 1, num_obs)
    return norm.logpdf(x, theta, 1.0).sum(axis=-1)


def gaussian_proposal(x, num_samples, scale_factor, seed=0):
    """Draws from N(posterior mean, (scale_factor * posterior scale)^2) and returns samples and log_q."""
    mean, scale = analytic_posterior(x)
    mean, scale = mean[:, None], scale_factor * scale
    theta = np.random.default_rng(seed).normal(mean, scale, size=(x.shape[0], num_samples))
    return {"theta": theta[..., None]}, norm.logpdf(theta, mean, scale)


@pytest.fixture()
def observed():
    rng = np.random.default_rng(1)
    theta = rng.normal(0.0, PRIOR_SCALE, size=(8, 1))
    return {"x": rng.normal(theta, 1.0, size=(8, NUM_OBS))}


def test_pareto_smooth_matches_arviz():
    from arviz_stats.base import array_stats

    log_ratios = np.random.default_rng(0).standard_t(df=3, size=(4, 1000))

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios)
    expected_log_weights, expected_k = array_stats.psislw(-log_ratios, axis=-1)

    np.testing.assert_allclose(log_weights, expected_log_weights)
    np.testing.assert_allclose(pareto_k, expected_k)


def test_pareto_smooth_shapes_and_normalization():
    log_ratios = np.random.default_rng(0).normal(size=(2, 3, 500))

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios)

    assert log_weights.shape == (2, 3, 500)
    assert pareto_k.shape == (2, 3)
    np.testing.assert_allclose(logsumexp(log_weights, axis=-1), 0.0, atol=1e-10)

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios[0, 0])
    assert log_weights.shape == (500,)
    assert pareto_k.shape == ()


def test_pareto_smooth_constant_ratios():
    log_ratios = np.zeros((2, 100))
    log_ratios[1] = np.random.default_rng(0).normal(size=100)

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios)

    np.testing.assert_allclose(log_weights[0], -np.log(100))
    assert pareto_k[0] == -np.inf
    assert np.isfinite(pareto_k[1])


def test_pareto_smooth_zero_target_density():
    log_ratios = np.random.default_rng(0).normal(size=200)
    log_ratios[3] = -np.inf

    log_weights, _ = bf.correction.pareto_smooth(log_ratios)

    assert log_weights[3] == -np.inf
    np.testing.assert_allclose(logsumexp(log_weights), 0.0, atol=1e-10)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_pareto_smooth_rejects_invalid_ratios(bad_value):
    log_ratios = np.zeros(100)
    log_ratios[0] = bad_value

    with pytest.raises(ValueError):
        bf.correction.pareto_smooth(log_ratios)


@pytest.mark.parametrize("num_samples", [10, 25])
def test_pareto_smooth_too_few_draws(num_samples):
    # the tail is too short to fit: weights stay unsmoothed and k flags them as unreliable
    log_ratios = np.random.default_rng(0).standard_cauchy(size=(2, num_samples))

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios)

    assert np.all(np.isnan(pareto_k) | (pareto_k == np.inf))
    np.testing.assert_allclose(logsumexp(log_weights, axis=-1), 0.0, atol=1e-10)


def test_pareto_smooth_tied_tail():
    log_ratios = np.r_[np.random.default_rng(0).normal(size=900), np.full(100, 5.0)]

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios)

    assert pareto_k == np.inf
    np.testing.assert_allclose(log_weights, log_ratios - logsumexp(log_ratios))


def test_pareto_smooth_all_zero_target_density():
    log_ratios = np.zeros((2, 100))
    log_ratios[1] = -np.inf

    with pytest.raises(ValueError, match="-inf"):
        bf.correction.pareto_smooth(log_ratios)


@pytest.mark.parametrize("target_scale", [1.5, 3.0])
def test_pareto_k_matches_theory(target_scale):
    # proposal N(0, 1), target N(0, s^2) with s > 1: the importance ratios have a Pareto tail with k = 1 - 1 / s^2
    theta = np.random.default_rng(0).normal(size=(20, 20_000))
    log_ratios = norm.logpdf(theta, 0.0, target_scale) - norm.logpdf(theta)

    _, pareto_k = bf.correction.pareto_smooth(log_ratios)

    assert abs(np.mean(pareto_k) - (1 - 1 / target_scale**2)) < 0.1


def test_psis_correction_exact_proposal(observed):
    samples, log_q = gaussian_proposal(observed["x"], 1000, scale_factor=1.0)

    # the log ratios are constant only up to round-off, which must not trigger the reliability warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = bf.correction.psis_correction(
            samples, log_prior, log_likelihood, conditions=observed, log_q=log_q, num_resamples=0
        )

    np.testing.assert_array_equal(result["pareto_k"], -np.inf)

    # the proposal equals the posterior, so all weights are equal up to round-off
    np.testing.assert_allclose(result["log_weights"], -np.log(1000), atol=1e-6)
    np.testing.assert_allclose(result["ess"], 1000, rtol=1e-6)


def test_psis_correction_recovers_posterior(observed):
    samples, log_q = gaussian_proposal(observed["x"], 4000, scale_factor=1.5)
    mean, scale = analytic_posterior(observed["x"])

    result = bf.correction.psis_correction(samples, log_prior, log_likelihood, conditions=observed, log_q=log_q)

    weights = np.exp(result["log_weights"])
    theta = samples["theta"][..., 0]
    weighted_mean = np.sum(weights * theta, axis=-1)
    weighted_var = np.sum(weights * (theta - weighted_mean[:, None]) ** 2, axis=-1)

    # a wider proposal has bounded weights, so PSIS is reliable
    assert np.all(result["pareto_k"] < 0.5)
    assert np.all(np.abs(weighted_mean - mean) <= 5 * scale / np.sqrt(result["ess"]))
    np.testing.assert_allclose(np.sqrt(weighted_var), scale, rtol=0.1)

    # resampled draws approximate the posterior
    resampled = result["samples"]["theta"]
    assert resampled.shape == (8, 4000, 1)
    assert np.all(np.abs(resampled[..., 0].mean(axis=-1) - mean) <= 6 * scale / np.sqrt(result["ess"]))


def test_psis_correction_warns_for_narrow_proposal(observed):
    samples, log_q = gaussian_proposal(observed["x"], 4000, scale_factor=0.5)

    with pytest.warns(UserWarning, match="Pareto k"):
        result = bf.correction.psis_correction(samples, log_prior, log_likelihood, conditions=observed, log_q=log_q)

    # theory: target scale / proposal scale = 2, so k = 1 - 1 / 4 = 0.75
    assert np.mean(result["pareto_k"]) > 0.6


def test_psis_correction_resampling_is_seeded(observed):
    samples, log_q = gaussian_proposal(observed["x"], 500, scale_factor=1.5)
    kwargs = dict(conditions=observed, log_q=log_q, num_resamples=100)

    a = bf.correction.psis_correction(samples, log_prior, log_likelihood, seed=42, **kwargs)
    b = bf.correction.psis_correction(samples, log_prior, log_likelihood, seed=42, **kwargs)

    assert a["samples"]["theta"].shape == (8, 100, 1)
    np.testing.assert_array_equal(a["samples"]["theta"], b["samples"]["theta"])


def test_psis_correction_invalid_arguments(observed):
    samples, log_q = gaussian_proposal(observed["x"], 100, scale_factor=1.5)

    # neither log_q nor estimator
    with pytest.raises(ValueError):
        bf.correction.psis_correction(samples, log_prior, log_likelihood, conditions=observed)

    # both log_q and estimator
    with pytest.raises(ValueError):
        bf.correction.psis_correction(
            samples, log_prior, log_likelihood, conditions=observed, log_q=log_q, estimator=object()
        )

    # log_q of wrong shape
    with pytest.raises(ValueError):
        bf.correction.psis_correction(samples, log_prior, log_likelihood, conditions=observed, log_q=log_q[:, :50])


def test_missing_arviz_stats_raises_helpful_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "arviz_stats", None)
    monkeypatch.setitem(sys.modules, "arviz_stats.base", None)

    with pytest.raises(ImportError, match="pip install arviz-stats"):
        bf.correction.pareto_smooth(np.zeros(100))


def test_estimator_log_q_is_normalized_density_over_constrained_parameter():
    """log_q from the estimator must include the adapter's Jacobian, or the importance weights are biased."""
    from scipy.stats import lognorm

    def prior():
        return dict(sigma=np.random.lognormal(0.0, 0.5, size=1))

    def likelihood(sigma):
        return dict(x=np.random.normal(0.0, sigma, size=10))

    simulator = bf.make_simulator([prior, likelihood])
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .constrain("sigma", lower=0)
        .rename("sigma", "inference_variables")
        .rename("x", "inference_conditions")
    )
    workflow = bf.BasicWorkflow(
        simulator=simulator, adapter=adapter, inference_network=bf.networks.CouplingFlow(depth=2)
    )
    workflow.fit_online(epochs=1, num_batches_per_epoch=5, batch_size=32, verbose=0)
    conditions = {"x": simulator.sample(2)["x"]}

    # the density over the constrained parameter integrates to one
    grid = np.linspace(1e-4, 15.0, 30_001)[:, None]
    log_q = workflow.estimator.log_prob({"x": np.repeat(conditions["x"][:1], len(grid), axis=0), "sigma": grid})
    np.testing.assert_allclose(np.trapezoid(np.exp(log_q), grid[:, 0]), 1.0, atol=1e-3)

    # psis_correction evaluates the estimator per data set, same as calling log_prob manually
    samples = workflow.estimator.sample(conditions=conditions, num_samples=50)

    def log_prior_sigma(samples):
        return lognorm.logpdf(samples["sigma"][..., 0], 0.5)

    def log_likelihood_sigma(samples, conditions):
        return norm.logpdf(conditions["x"][:, None, :], 0.0, samples["sigma"]).sum(axis=-1)

    result = bf.correction.psis_correction(
        samples,
        log_prior_sigma,
        log_likelihood_sigma,
        conditions=conditions,
        estimator=workflow.estimator,
        num_resamples=0,
    )
    manual_log_q = np.stack(
        [
            workflow.estimator.log_prob(
                {"x": np.repeat(conditions["x"][j : j + 1], 50, axis=0), "sigma": samples["sigma"][j]}
            )
            for j in range(2)
        ]
    )
    expected, _ = bf.correction.pareto_smooth(
        log_prior_sigma(samples) + log_likelihood_sigma(samples, conditions) - manual_log_q
    )
    np.testing.assert_allclose(result["log_weights"], expected, rtol=1e-5, atol=1e-6)


def test_psis_correction_fixes_undertrained_estimator():
    """End-to-end: PSIS recovers the analytic posterior from a briefly trained, too-wide approximation."""

    def prior():
        return dict(theta=np.random.normal(0.0, PRIOR_SCALE, size=1))

    def likelihood(theta):
        return dict(x=np.random.normal(theta, 1.0, size=NUM_OBS))

    simulator = bf.make_simulator([prior, likelihood])
    adapter = (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .rename("x", "inference_conditions")
    )
    workflow = bf.BasicWorkflow(
        simulator=simulator, adapter=adapter, inference_network=bf.networks.CouplingFlow(depth=2)
    )
    workflow.fit_online(epochs=3, num_batches_per_epoch=50, batch_size=64, verbose=0)

    conditions = {"x": simulator.sample(20)["x"]}
    mean, scale = analytic_posterior(conditions["x"])
    samples = workflow.estimator.sample(conditions=conditions, num_samples=2000)

    result = bf.correction.psis_correction(
        samples, log_prior, log_likelihood, conditions=conditions, estimator=workflow.estimator, seed=0
    )

    corrected = result["samples"]["theta"][..., 0]
    assert np.all(result["pareto_k"] < 0.7)
    assert np.mean(np.abs(corrected.mean(axis=-1) - mean)) < 0.2 * scale
    assert 0.9 < np.mean(corrected.std(axis=-1)) / scale < 1.1


def test_psis_correction_warns_for_unfittable_tail(observed):
    # 25 draws are too few to fit the tail, which must not pass as reliable
    samples, log_q = gaussian_proposal(observed["x"], 25, scale_factor=0.5)

    with pytest.warns(UserWarning, match="Pareto k"):
        result = bf.correction.psis_correction(samples, log_prior, log_likelihood, conditions=observed, log_q=log_q)

    assert np.all(np.isnan(result["pareto_k"]) | (result["pareto_k"] > 0.5))


def test_psis_correction_warns_when_proposal_leaks_out_of_prior_support(observed):
    samples, log_q = gaussian_proposal(observed["x"], 1000, scale_factor=1.5)

    def truncated_log_prior(samples):
        # zero prior density for most draws
        log_density = log_prior(samples)
        theta = samples["theta"][..., 0]
        cutoff = theta.mean(axis=-1, keepdims=True) + 1.75 * theta.std(axis=-1, keepdims=True)
        return np.where(theta > cutoff, log_density, -np.inf)

    with pytest.warns(UserWarning, match="Pareto k"):
        bf.correction.psis_correction(samples, truncated_log_prior, log_likelihood, conditions=observed, log_q=log_q)


def test_psis_correction_ess_scales_with_r_eff(observed):
    samples, log_q = gaussian_proposal(observed["x"], 1000, scale_factor=1.0)
    kwargs = dict(conditions=observed, log_q=log_q, num_resamples=0)

    result = bf.correction.psis_correction(samples, log_prior, log_likelihood, r_eff=0.5, **kwargs)

    np.testing.assert_allclose(result["ess"], 500, rtol=1e-6)


def test_psis_correction_invalid_samples_and_resamples(observed):
    samples, log_q = gaussian_proposal(observed["x"], 100, scale_factor=1.5)

    with pytest.raises(ValueError, match="return_summaries"):
        bf.correction.psis_correction(
            samples | {"_summaries": np.zeros((8, 4))},
            log_prior,
            log_likelihood,
            conditions=observed,
            log_q=log_q,
        )

    with pytest.raises(ValueError):
        bf.correction.psis_correction(
            samples, log_prior, log_likelihood, conditions=observed, log_q=log_q, num_resamples=-1
        )


def test_psis_correction_resampling_multidimensional_samples(observed):
    samples, log_q = gaussian_proposal(observed["x"], 200, scale_factor=1.5)
    # an extra (2, 3) variable whose entries encode the draw index, to check that resampling stays aligned
    index = np.broadcast_to(np.arange(200)[None, :, None, None], (8, 200, 2, 3)).astype(float)
    samples = samples | {"index": index}

    result = bf.correction.psis_correction(
        samples, log_prior, log_likelihood, conditions=observed, log_q=log_q, num_resamples=50, seed=0
    )

    resampled = result["samples"]
    assert resampled["index"].shape == (8, 50, 2, 3)
    drawn = resampled["index"][:, :, 0, 0].astype(int)
    np.testing.assert_array_equal(
        np.take_along_axis(samples["theta"][..., 0], drawn, axis=1), resampled["theta"][..., 0]
    )


def test_pareto_smooth_float32_round_off_is_constant():
    # log densities computed in float32 (e.g., by an estimator) are constant only up to ~1e-6 relative noise
    log_ratios = 10.0 + np.random.default_rng(0).normal(scale=1e-6, size=(3, 1000))

    log_weights, pareto_k = bf.correction.pareto_smooth(log_ratios)

    np.testing.assert_array_equal(pareto_k, -np.inf)
    np.testing.assert_allclose(log_weights, -np.log(1000), atol=1e-5)


def test_pareto_smooth_single_positive_draw_is_not_reliable():
    # all weight on one draw must not be reported as constant (k = -inf)
    log_ratios = np.r_[0.3, np.full(999, -np.inf)]

    _, pareto_k = bf.correction.pareto_smooth(log_ratios)

    assert not pareto_k == -np.inf


def test_pareto_smooth_invalid_arguments():
    with pytest.raises(ValueError):
        bf.correction.pareto_smooth(np.zeros((3, 1)))

    with pytest.raises(ValueError):
        bf.correction.pareto_smooth(np.zeros(100), r_eff=0.0)


def test_psis_correction_warns_for_single_positive_draw(observed):
    samples, log_q = gaussian_proposal(observed["x"], 1000, scale_factor=1.5)

    def degenerate_log_prior(samples):
        log_density = np.full(samples["theta"].shape[:2], -np.inf)
        log_density[:, 0] = 0.0
        return log_density

    with pytest.warns(UserWarning, match="Pareto k"):
        bf.correction.psis_correction(samples, degenerate_log_prior, log_likelihood, conditions=observed, log_q=log_q)


def test_psis_correction_degenerate_samples(observed):
    _, log_q = gaussian_proposal(observed["x"], 100, scale_factor=1.5)

    for samples in [{}, {"theta": np.zeros(100)}, {"theta": np.zeros((0, 100, 1))}]:
        with pytest.raises(ValueError):
            bf.correction.psis_correction(samples, log_prior, log_likelihood, conditions=observed, log_q=log_q)


def test_psis_correction_estimator_with_broadcast_context():
    """Conditions without a data set axis, e.g., a scalar broadcast by the adapter, are passed unchanged."""

    def prior():
        return dict(theta=np.random.normal(0.0, PRIOR_SCALE, size=1))

    def likelihood(theta):
        return dict(x=np.random.normal(theta, 1.0, size=NUM_OBS))

    simulator = bf.make_simulator([prior, likelihood])
    adapter = (
        bf.Adapter()
        .broadcast("N", to="x")
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .concatenate(["x", "N"], into="inference_conditions")
    )
    workflow = bf.BasicWorkflow(
        simulator=simulator, adapter=adapter, inference_network=bf.networks.CouplingFlow(depth=1)
    )
    workflow.fit_offline(simulator.sample(256) | {"N": float(NUM_OBS)}, epochs=1, batch_size=64, verbose=0)

    conditions = {"x": simulator.sample(3)["x"], "N": float(NUM_OBS)}
    samples = workflow.estimator.sample(conditions=conditions, num_samples=50)

    with warnings.catch_warnings():
        # an untrained flow may give unreliable weights; only the evaluation of log_q is tested here
        warnings.simplefilter("ignore", UserWarning)
        result = bf.correction.psis_correction(
            samples, log_prior, log_likelihood, conditions=conditions, estimator=workflow.estimator, num_resamples=0
        )
        manual_log_q = np.stack(
            [
                workflow.estimator.log_prob(
                    {"x": np.repeat(conditions["x"][j : j + 1], 50, axis=0), "N": conditions["N"]}
                    | {"theta": samples["theta"][j]}
                )
                for j in range(3)
            ]
        )
        expected, _ = bf.correction.pareto_smooth(
            log_prior(samples) + log_likelihood(samples, conditions) - manual_log_q
        )

    np.testing.assert_allclose(result["log_weights"], expected, rtol=1e-5, atol=1e-6)


def test_psis_correction_single_dataset_matches_batched(observed):
    """A 1-D log_q denotes a single data set: callables see the inputs as passed, results lose the batch axis."""
    samples, log_q = gaussian_proposal(observed["x"][:1], 2000, scale_factor=1.5)
    batched = bf.correction.psis_correction(
        samples, log_prior, log_likelihood, conditions={"x": observed["x"][:1]}, log_q=log_q, seed=0
    )

    seen = {}

    def single_log_prior(s):
        seen["prior"] = s["theta"].shape
        return norm.logpdf(s["theta"][:, 0], 0.0, PRIOR_SCALE)

    def single_log_likelihood(s, c):
        seen["likelihood"] = (s["theta"].shape, c["x"].shape)
        return norm.logpdf(c["x"][None, :], s["theta"], 1.0).sum(axis=-1)

    single = bf.correction.psis_correction(
        {"theta": samples["theta"][0]},
        single_log_prior,
        single_log_likelihood,
        conditions={"x": observed["x"][0]},
        log_q=log_q[0],
        seed=0,
    )

    assert seen == {"prior": (2000, 1), "likelihood": ((2000, 1), (NUM_OBS,))}
    np.testing.assert_allclose(single["log_weights"], batched["log_weights"][0])
    np.testing.assert_allclose(single["pareto_k"], batched["pareto_k"][0])
    np.testing.assert_allclose(single["ess"], batched["ess"][0])
    np.testing.assert_array_equal(single["samples"]["theta"], batched["samples"]["theta"][0])
    assert np.ndim(single["pareto_k"]) == 0
    assert single["samples"]["theta"].shape == (2000, 1)


def test_psis_correction_single_dataset_shape_errors(observed):
    samples, log_q = gaussian_proposal(observed["x"][:1], 100, scale_factor=1.5)
    unbatched = {"theta": samples["theta"][0]}

    # unbatched samples with a 2-D log_q: the error must point to the single data set usage
    with pytest.raises(ValueError, match="For a single data set"):
        bf.correction.psis_correction(unbatched, log_prior, log_likelihood, conditions=observed, log_q=log_q)

    # single data set, but the callables return a batched shape
    with pytest.raises(ValueError, match=r"log_prior to have shape \(num_samples,\)"):
        bf.correction.psis_correction(
            unbatched,
            lambda s: np.zeros((1, 100)),
            lambda s, c: np.zeros(100),
            conditions={"x": observed["x"][0]},
            log_q=log_q[0],
        )

    # mismatched number of draws between samples and log_q
    with pytest.raises(ValueError, match="log_q"):
        bf.correction.psis_correction(
            unbatched,
            lambda s: np.zeros(100),
            lambda s, c: np.zeros(100),
            conditions={"x": observed["x"][0]},
            log_q=log_q[0, :50],
        )


def test_psis_correction_shape_hints_name_the_cause(observed):
    samples, log_q = gaussian_proposal(observed["x"][:1], 100, scale_factor=1.5)

    # half-batched: batched samples with a 1-D log_q
    with pytest.raises(ValueError, match="A 1-D log_q denotes a single data set"):
        bf.correction.psis_correction(
            samples, lambda s: np.zeros(1), lambda s, c: np.zeros(1), conditions=observed, log_q=log_q[0]
        )

    # draws read as data sets: one draw per data set
    with pytest.raises(ValueError, match="at least two draws"):
        bf.correction.psis_correction(
            {"theta": samples["theta"][0][:, None]},
            log_prior,
            log_likelihood,
            conditions=observed,
            log_q=log_q[0][:, None],
        )


def test_psis_correction_estimator_hint_does_not_suggest_log_q(observed):
    class DummyEstimator(bf.estimators.BasicEstimator):
        def __init__(self):
            pass

    samples, _ = gaussian_proposal(observed["x"][:1], 100, scale_factor=1.5)

    with pytest.raises(ValueError, match="add a leading axis of length 1 to all samples and conditions") as info:
        bf.correction.psis_correction(
            {"theta": samples["theta"][0, :, 0]},
            log_prior,
            log_likelihood,
            conditions=observed,
            estimator=DummyEstimator(),
        )
    assert "1-D log_q" not in str(info.value)


def test_psis_correction_callable_errors_explain_calling_convention(observed):
    samples, log_q = gaussian_proposal(observed["x"], 100, scale_factor=1.5)

    def swapped_log_likelihood(conditions, samples):
        return log_likelihood(samples, conditions)

    with pytest.raises(ValueError, match=r"swapped.*log_likelihood\(samples, conditions\)"):
        bf.correction.psis_correction(samples, log_prior, swapped_log_likelihood, conditions=observed, log_q=log_q)

    with pytest.raises(ValueError, match="log_prior receives only the samples"):
        bf.correction.psis_correction(samples, lambda s: s["x"], log_likelihood, conditions=observed, log_q=log_q)

    # other errors keep their type and gain a note on the calling convention
    with pytest.raises(TypeError) as info:
        bf.correction.psis_correction(
            samples, lambda s, c: log_prior(s), log_likelihood, conditions=observed, log_q=log_q
        )
    assert any("log_prior is called as `log_prior(samples)`" in note for note in info.value.__notes__)

    with pytest.raises(KeyError) as info:
        bf.correction.psis_correction(samples, lambda s: s["mu"], log_likelihood, conditions=observed, log_q=log_q)
    assert any("samples of shapes {'theta': (8, 100, 1)}" in note for note in info.value.__notes__)


def test_psis_correction_log_density_errors_name_the_cause(observed):
    samples, log_q = gaussian_proposal(observed["x"], 100, scale_factor=1.5)

    def per_observation_log_likelihood(samples, conditions):
        return norm.logpdf(conditions["x"][:, None, :], samples["theta"], 1.0)

    with pytest.raises(ValueError, match="sum over the last axis"):
        bf.correction.psis_correction(
            samples, log_prior, per_observation_log_likelihood, conditions=observed, log_q=log_q
        )

    with pytest.raises(ValueError, match="not a sum over draws"):
        bf.correction.psis_correction(
            samples, lambda s: np.sum(log_prior(s)), log_likelihood, conditions=observed, log_q=log_q
        )

    with pytest.raises(ValueError, match=r"log_prior returned nan or \+inf for 8 of 800 draws"):
        bf.correction.psis_correction(
            samples,
            lambda s: np.where(np.arange(100) == 0, np.nan, log_prior(s)),
            log_likelihood,
            conditions=observed,
            log_q=log_q,
        )

    with pytest.raises(ValueError, match="Expected log_q to be finite"):
        bf.correction.psis_correction(
            samples, log_prior, log_likelihood, conditions=observed, log_q=np.where(log_q > -1, -np.inf, log_q)
        )
