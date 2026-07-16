import keras
import pytest


@pytest.fixture()
def reference(batch_size, feature_size):
    return keras.random.uniform((batch_size, feature_size))


@pytest.fixture()
def median_score():
    from bayesflow.scoring_rules import MedianScore

    return MedianScore()


@pytest.fixture()
def mean_score():
    from bayesflow.scoring_rules import MeanScore

    return MeanScore()


@pytest.fixture()
def normed_diff_score():
    from bayesflow.scoring_rules import NormedDifferenceScore

    return NormedDifferenceScore(k=3)


@pytest.fixture(scope="function")
def quantile_score():
    from bayesflow.scoring_rules import QuantileScore

    return QuantileScore()


@pytest.fixture()
def multivariate_normal_score():
    from bayesflow.scoring_rules import MvNormalScore

    return MvNormalScore()


@pytest.fixture()
def mixture_of_multivariate_normal_scores():
    from bayesflow.scoring_rules import MvNormalScore, MixtureScore

    return MixtureScore(mvn1=MvNormalScore(), mvn2=MvNormalScore())


# --- Model comparison scoring rules ---


@pytest.fixture()
def cross_entropy_score():
    from bayesflow.scoring_rules import CrossEntropyScore

    return CrossEntropyScore()


@pytest.fixture()
def brier_score():
    from bayesflow.scoring_rules import BrierScore

    return BrierScore()


@pytest.fixture()
def polynomial_score():
    from bayesflow.scoring_rules import PolynomialScore

    return PolynomialScore(alpha=3.0)


@pytest.fixture()
def exponential_score():
    from bayesflow.scoring_rules import ExponentialScore

    return ExponentialScore()


@pytest.fixture()
def leaky_exponential_score():
    from bayesflow.links import Leaky
    from bayesflow.scoring_rules import ExponentialScore

    return ExponentialScore(links={"logits": Leaky(power=2.0)})


@pytest.fixture()
def logistic_score():
    from bayesflow.scoring_rules import LogisticScore

    return LogisticScore()


@pytest.fixture()
def power_logistic_score():
    from bayesflow.scoring_rules import LogisticScore

    return LogisticScore(alpha=1.0)


@pytest.fixture(
    params=[
        "median_score",
        "mean_score",
        "normed_diff_score",
        "quantile_score",
        "multivariate_normal_score",
        "mixture_of_multivariate_normal_scores",
        "cross_entropy_score",
        "brier_score",
        "polynomial_score",
        "exponential_score",
        "leaky_exponential_score",
        "logistic_score",
        "power_logistic_score",
    ],
    scope="function",
)
def scoring_rule(request):
    print("initialize scoring rule in test_scoring_rules")
    return request.getfixturevalue(request.param)
