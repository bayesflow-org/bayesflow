import keras
import pytest


@pytest.fixture()
def reference(batch_size, feature_size):
    return keras.random.uniform((batch_size, feature_size))


@pytest.fixture()
def median_score():
    from bayesflow.scoring_rules import MedianScoringRule

    return MedianScoringRule()


@pytest.fixture()
def mean_score():
    from bayesflow.scoring_rules import MeanScoringRule

    return MeanScoringRule()


@pytest.fixture()
def normed_diff_score():
    from bayesflow.scoring_rules import NormedDifferenceScoringRule

    return NormedDifferenceScoringRule(k=3)


@pytest.fixture(scope="function")
def quantile_score():
    from bayesflow.scoring_rules import QuantileScoringRule

    return QuantileScoringRule()


@pytest.fixture()
def multivariate_normal_score():
    from bayesflow.scoring_rules import MvNormalScoringRule

    return MvNormalScoringRule()


@pytest.fixture(
    params=["median_score", "mean_score", "normed_diff_score", "quantile_score", "multivariate_normal_score"],
    scope="function",
)
def scoring_rule(request):
    print("initialize scoring rule in test_scoring_rules")
    return request.getfixturevalue(request.param)
