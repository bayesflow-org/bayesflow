import pytest


@pytest.fixture()
def summary_network():
    from bayesflow.networks import DeepSet

    return DeepSet(summary_dim=2)


@pytest.fixture()
def inference_network():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(subnet="mlp", depth=2, subnet_kwargs=dict(widths=(32, 32)))


@pytest.fixture(
    params=[
        "all",
        None,
        "inference_variables",
        "summary_variables",
        ("inference_variables", "summary_variables", "inference_conditions"),
    ]
)
def standardize(request):
    return request.param


@pytest.fixture()
def approximator(adapter, inference_network, summary_network, standardize):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=inference_network,
        summary_network=summary_network,
        standardize=standardize,
    )


@pytest.fixture()
def simulator():
    from tests.utils.normal_simulator import NormalSimulator

    return NormalSimulator()


@pytest.fixture
def adapter():
    from bayesflow import Adapter

    adapter = (
        Adapter()
        .create_default(["mean"])
        .rename("std", "inference_conditions")
        .rename("x", "summary_variables")
        .expand_dims("summary_variables", axis=-1)
    )
    return adapter
