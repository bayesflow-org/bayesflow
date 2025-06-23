import pytest


@pytest.fixture()
def batch_size():
    return 8


@pytest.fixture(params=["single_parametric", "multiple_parametric"])
def point_inference_network(request):
    match request.param:
        case "single_parametric":
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

        case "multiple_parametric":
            from bayesflow.networks import PointInferenceNetwork
            from bayesflow.scores import MultivariateNormalScore

            return PointInferenceNetwork(
                scores=dict(
                    mvn1=MultivariateNormalScore(),
                    mvn2=MultivariateNormalScore(),
                ),
            )
        case _:
            raise ValueError(f"Invalid request parameter for point_inference_network: {request.param}")


@pytest.fixture
def approximator(adapter, point_inference_network, summary_network, standardize):
    from bayesflow.approximators import PointApproximator

    return PointApproximator(
        adapter=adapter,
        inference_network=point_inference_network,
        summary_network=summary_network,
        standardize=standardize,
    )
