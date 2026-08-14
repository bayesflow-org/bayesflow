from bayesflow.experimental.self_consistency import SelfConsistencyLoss


def test_attach(posterior_approximator, analytic_prior, analytic_likelihood, shared_adapter):
    sc = SelfConsistencyLoss(num_samples=4)
    approximators = {
        "prior": analytic_prior,
        "likelihood": analytic_likelihood,
        "posterior": posterior_approximator,
    }
    sc.attach(approximators, shared_adapter)

    assert sc.prior is analytic_prior
    assert sc.likelihood is analytic_likelihood
    assert sc.posterior is posterior_approximator
    assert sc.adapter is shared_adapter
