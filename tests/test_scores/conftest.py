import keras
import pytest


@pytest.fixture()
def reference(batch_size, feature_size):
    return keras.random.uniform((batch_size, feature_size))


@pytest.fixture()
def median_score():
    from bayesflow.scores import MedianScore

    return MedianScore()


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

    return QuantileScore()


@pytest.fixture()
def multivariate_normal_score():
    from bayesflow.scores import MultivariateNormalScore

    return MultivariateNormalScore()


@pytest.fixture(
    params=["median_score", "mean_score", "normed_diff_score", "quantile_score", "multivariate_normal_score"],
    scope="function",
)
def scoring_rule(request):
    print("initialize scoring rule in test_scores")
    return request.getfixturevalue(request.param)
