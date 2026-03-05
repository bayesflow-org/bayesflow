import pytest
import numpy as np


@pytest.fixture()
def median_score():
    from bayesflow.scoring_rules import MedianScore

    return MedianScore()


@pytest.fixture()
def median_score_subnet():
    from bayesflow.scoring_rules import MedianScore

    return MedianScore(subnets=dict(value="mlp"))


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

    return QuantileScore(q=[0.2, 0.3, 0.4, 0.5, 0.7])


@pytest.fixture()
def multivariate_normal_score():
    from bayesflow.scoring_rules import MvNormalScore

    return MvNormalScore()


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
def scoring_rule_network(scoring_rule):
    from bayesflow.networks import ScoringRuleNetwork

    return ScoringRuleNetwork(
        scoring_rules=dict(dummy_name=scoring_rule),
    )


@pytest.fixture(scope="function")
def quantile_scoring_rule_network():
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import QuantileScore

    return ScoringRuleNetwork(
        scoring_rules=dict(quantiles=QuantileScore(q=np.array([0.1, 0.4, 0.5, 0.7]), subnets=dict(value="mlp"))),
    )
