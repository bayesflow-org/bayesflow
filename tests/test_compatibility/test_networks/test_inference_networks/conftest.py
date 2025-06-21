import pytest

from bayesflow.networks import MLP


@pytest.fixture
def inference_network(request):
    name, kwargs = request.param
    from bayesflow.utils.dispatch import find_inference_network

    try:
        return find_inference_network(name, **kwargs)
    except ValueError:
        # network not yet in find_inference_network
        pass
    match name:
        case "diffusion_model":
            from bayesflow.experimental import DiffusionModel

            return DiffusionModel(**kwargs)
        case "free_form_flow":
            from bayesflow.experimental import FreeFormFlow

            return FreeFormFlow(**kwargs)
        case "point_inference_network":
            from bayesflow.networks import PointInferenceNetwork
            from bayesflow.scores import MeanScore, MedianScore, QuantileScore, MultivariateNormalScore

            return PointInferenceNetwork(
                scores=dict(
                    mean=MeanScore(subnets=dict(value=MLP([16, 8]))),
                    median=MedianScore(subnets=dict(value=MLP([16, 8]))),
                    quantiles=QuantileScore(subnets=dict(value=MLP([16, 8]))),
                    mvn=MultivariateNormalScore(subnets=dict(mean=MLP([16, 8]), covariance=MLP([16, 8]))),
                ),
                **kwargs,
            )
        case _:
            raise ValueError(f"Invalid request parameter for inference_network: {name}")
