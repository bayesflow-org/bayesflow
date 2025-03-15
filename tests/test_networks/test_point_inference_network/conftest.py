import pytest
import numpy as np


@pytest.fixture()
def median_score():
    from bayesflow.scores import MedianScore

    return MedianScore()


@pytest.fixture()
def median_score_subnet():
    from bayesflow.scores import MedianScore

    return MedianScore(subnets=dict(value="mlp"))


@pytest.fixture()
def mean_score():
    from bayesflow.scores import MeanScore

    return MeanScore()


@pytest.fixture()
def normed_diff_score():
    from bayesflow.scores import NormedDifferenceScore

    return NormedDifferenceScore(k=3)


@pytest.fixture(scope="function")
def quantile_score():
    from bayesflow.scores import QuantileScore

    return QuantileScore(q=[0.2, 0.3, 0.4, 0.5, 0.7])


@pytest.fixture()
def multivariate_normal_score():
    from bayesflow.scores import MultivariateNormalScore

    return MultivariateNormalScore()


@pytest.fixture(
    params=[
        "median_score",
        "median_score_subnet",
        "mean_score",
        "normed_diff_score",
        "quantile_score",
        "multivariate_normal_score",
    ],
)
def scoring_rule(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="function")
def point_inference_network(scoring_rule):
    from bayesflow.networks import PointInferenceNetwork

    return PointInferenceNetwork(
        scores=dict(dummy_name=scoring_rule),
    )


@pytest.fixture(scope="function")
def quantile_point_inference_network():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import QuantileScore

    return PointInferenceNetwork(
        scores=dict(quantiles=QuantileScore(q=np.array([0.1, 0.4, 0.5, 0.7]), subnets=dict(value="mlp"))),
    )
