from bayesflow.experimental.self_consistency import SelfConsistencyLoss


def test_attach(posterior_estimator, analytic_prior, analytic_likelihood, shared_adapter):
    sc = SelfConsistencyLoss(num_samples=4)
    estimators = {
        "prior": analytic_prior,
        "likelihood": analytic_likelihood,
        "posterior": posterior_estimator,
    }
    sc.attach(estimators, shared_adapter)

    assert sc.prior is analytic_prior
    assert sc.likelihood is analytic_likelihood
    assert sc.posterior is posterior_estimator
    assert sc.adapter is shared_adapter
