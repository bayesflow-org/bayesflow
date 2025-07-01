import pytest


@pytest.fixture
def approximator(adapter, inference_network, summary_network, standardize):
    from bayesflow.approximators import ContinuousApproximator

    return ContinuousApproximator(
        adapter=adapter, inference_network=inference_network, summary_network=summary_network, standardize=standardize
    )
