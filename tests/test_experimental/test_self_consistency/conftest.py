from bayesflow.experimental.self_consistency import SemiSupervisedEstimator, MultiDataset
import numpy as np
import pytest
import keras
from tensorflow_probability import distributions as tfd

from bayesflow.datasets import OfflineDataset
from bayesflow.adapters import Adapter
from bayesflow.estimators import BasicEstimator
from bayesflow.networks import CouplingFlow
from bayesflow import make_simulator


@pytest.fixture()
def simulator():
    def sim_fn():
        tau = np.random.gamma(size=1, shape=2, scale=1)
        mu = np.random.normal(size=1, scale=1.0 / np.sqrt(tau))
        y = np.random.normal(size=10, loc=mu, scale=1.0 / np.sqrt(tau))
        return dict(mu=mu, tau=tau, y=y)

    return make_simulator(sim_fn)


@pytest.fixture()
def labeled_dataset(simulator):
    data = simulator.sample(32)
    return OfflineDataset(data=data, batch_size=8, adapter=Adapter())


@pytest.fixture()
def unlabeled_dataset(simulator):
    data = dict(y=simulator.sample(8)["y"])
    return OfflineDataset(data=data, batch_size=4, adapter=Adapter())


@pytest.fixture()
def dataset(labeled_dataset, unlabeled_dataset):
    return MultiDataset(labeled=labeled_dataset, unlabeled=unlabeled_dataset)


@pytest.fixture()
def posterior_adapter():
    return Adapter().rename("y", "inference_conditions").concatenate(["mu", "tau"], into="inference_variables")


@pytest.fixture()
def shared_adapter():
    return Adapter().log("tau")


@pytest.fixture()
def posterior_estimator(posterior_adapter):
    return BasicEstimator(
        inference_network=CouplingFlow(),
        adapter=posterior_adapter,
    )


@pytest.fixture()
def analytic_prior():
    @keras.saving.register_keras_serializable()
    def prior_log_prob(mu, tau, **kwargs):
        tau_log_prob = tfd.Gamma(concentration=2.0, rate=1.0).log_prob(tau)
        mu_log_prob = tfd.Normal(loc=0.0, scale=1.0 / keras.ops.sqrt(tau)).log_prob(mu)
        return mu_log_prob + tau_log_prob

    return prior_log_prob


@pytest.fixture()
def analytic_likelihood():
    @keras.saving.register_keras_serializable()
    def likelihood_log_prob(y, mu, tau, **kwargs):
        y_log_prob = tfd.Normal(loc=mu, scale=1.0 / keras.ops.sqrt(tau)).log_prob(y)
        return keras.ops.sum(y_log_prob, axis=1)

    return likelihood_log_prob


@pytest.fixture()
def sc_estimator(posterior_estimator, analytic_prior, analytic_likelihood, shared_adapter):
    return SemiSupervisedEstimator(
        estimators=dict(prior=analytic_prior, likelihood=analytic_likelihood, posterior=posterior_estimator),
        adapter=shared_adapter,
    )
