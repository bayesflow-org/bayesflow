import pytest


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(subnet="mlp", depth=2, subnet_kwargs=dict(widths=(32, 32)))


@pytest.fixture()
def basic_estimator(adapter, inference_network, summary_network):
    from bayesflow import BasicEstimator

    return BasicEstimator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )
