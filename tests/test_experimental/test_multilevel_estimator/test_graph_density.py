import keras
import numpy as np
import pytest
from scipy import stats

from bayesflow.experimental.multilevel_estimator.batches import tile_batch, to_batch
from bayesflow.experimental.multilevel_simulator import MultilevelSimulator
from bayesflow.experimental.multilevel_simulator.example_simulators.two_level import (
    locals_log_prob,
    sample_hypers,
    sample_locals,
)


def expected_log_probs(data):
    hypers, locals_, shared, y = data["hypers"], data["locals"], data["shared"], data["y"]
    datasets = len(hypers.indices["dataset"])

    def per_dataset(row_log_prob, node_samples):
        return np.bincount(node_samples.indices["dataset"], weights=row_log_prob.ravel(), minlength=datasets)

    hyper_mean, hyper_std = hypers.values["hyper_mean"], hypers.values["hyper_std"]
    hypers_lp = stats.norm.logpdf(hyper_mean) + stats.halfnorm.logpdf(hyper_std)
    locals_lp = stats.norm.logpdf(
        locals_.values["local_mean"], hyper_mean[locals_.indices["hypers"]], hyper_std[locals_.indices["hypers"]]
    )
    shared_lp = stats.halfnorm.logpdf(shared.values["shared_std"])
    local_mean = locals_.values["local_mean"][y.indices["locals"]]
    shared_std = shared.values["shared_std"][y.indices["shared"]]
    y_lp = stats.norm.logpdf(y.values["y"], local_mean, shared_std)

    prior = per_dataset(hypers_lp, hypers) + per_dataset(locals_lp, locals_) + per_dataset(shared_lp, shared)
    likelihood = per_dataset(y_lp, y)

    return prior, likelihood


def test_prior_and_likelihood_match_scipy(two_level_simulator):
    data = two_level_simulator.sample(3)
    batch = to_batch(data, two_level_simulator.graph.expand().invert())

    prior, likelihood = expected_log_probs(data)

    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(two_level_simulator.prior.log_prob(**batch)), prior, rtol=1e-5
    )
    np.testing.assert_allclose(
        keras.ops.convert_to_numpy(two_level_simulator.likelihood.log_prob(**batch)), likelihood, rtol=1e-5
    )


def test_log_prob_of_copies_matches_original(two_level_simulator):
    batch = to_batch(two_level_simulator.sample(2), two_level_simulator.graph.expand().invert())
    copies = 3

    for density in (two_level_simulator.prior, two_level_simulator.likelihood):
        log_prob = keras.ops.convert_to_numpy(density.log_prob(**batch))
        tiled = keras.ops.convert_to_numpy(density.log_prob(**tile_batch(batch, copies)))

        np.testing.assert_allclose(tiled, np.repeat(log_prob, copies), rtol=1e-5)


def test_missing_log_prob_fn_raises():
    simulator = MultilevelSimulator()
    simulator.add_node("hypers", sample_fn=sample_hypers)
    simulator.add_node("locals", sample_fn=sample_locals, log_prob_fn=locals_log_prob)
    simulator.add_edge("hypers", "locals")

    with pytest.raises(ValueError, match="hypers"):
        simulator.prior
