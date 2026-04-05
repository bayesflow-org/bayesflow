import pytest


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(subnet="mlp", depth=2, subnet_kwargs=dict(widths=(32, 32)))


@pytest.fixture()
def continuous_approximator(adapter, inference_network, summary_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
    )
