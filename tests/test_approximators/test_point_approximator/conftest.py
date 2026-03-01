import pytest


@pytest.fixture()
def point_inference_network():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import NormedDifferenceScore, QuantileScore, MultivariateNormalScore

    return PointInferenceNetwork(
        scores=dict(
            mean=NormedDifferenceScore(k=2),
            quantiles=QuantileScore(q=[0.1, 0.5, 0.9]),
            mvn=MultivariateNormalScore(),
        ),
        subnet="mlp",
        subnet_kwargs=dict(widths=(32, 32)),
    )


@pytest.fixture()
def point_inference_network_with_multiple_parametric_scores():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MultivariateNormalScore

    return PointInferenceNetwork(
        scores=dict(
            mvn1=MultivariateNormalScore(),
            mvn2=MultivariateNormalScore(),
        ),
    )


@pytest.fixture()
def point_approximator_with_single_parametric_score(adapter, point_inference_network, summary_network):
    from bayesflow import PointApproximator

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("point approximator does not support unconditional estimation")

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def point_approximator_with_multiple_parametric_scores(
    adapter, point_inference_network_with_multiple_parametric_scores, summary_network
):
    from bayesflow import PointApproximator

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("point approximator does not support unconditional estimation")

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network_with_multiple_parametric_scores,
        summary_network=summary_network,
    )


@pytest.fixture(
    params=["point_approximator_with_single_parametric_score", "point_approximator_with_multiple_parametric_scores"]
)
def point_approximator(request):
    return request.getfixturevalue(request.param)
