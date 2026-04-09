import pytest


@pytest.fixture()
def scoring_rule_inference_network():
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import NormedDifferenceScore, QuantileScore, MvNormalScore, MixtureScore

    return ScoringRuleNetwork(
        scoring_rules=dict(
            mean=NormedDifferenceScore(k=2),
            quantiles=QuantileScore(q=[0.1, 0.5, 0.9]),
            mvn=MvNormalScore(),
            mix=MixtureScore(mvn_c1=MvNormalScore(), mvn_c2=MvNormalScore()),
        ),
        subnet="mlp",
        subnet_kwargs=dict(widths=(8, 8)),
    )


@pytest.fixture()
def scoring_rule_inference_network_with_multiple_parametric_scores():
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import MvNormalScore

    return ScoringRuleNetwork(
        scoring_rules=dict(
            mvn1=MvNormalScore(),
            mvn2=MvNormalScore(),
        ),
    )


@pytest.fixture()
def scoring_rule_approximator_with_single_parametric_score(adapter, scoring_rule_inference_network, summary_network):
    from bayesflow import ScoringRuleApproximator

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("Scoring rule approximator does not support unconditional estimation")

    return ScoringRuleApproximator(
        adapter=adapter,
        inference_network=scoring_rule_inference_network,
        summary_network=summary_network,
    )


@pytest.fixture()
def scoring_rule_approximator_with_multiple_parametric_scores(
    adapter, scoring_rule_inference_network_with_multiple_parametric_scores, summary_network
):
    from bayesflow import ScoringRuleApproximator

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("Scoring rule approximator does not support unconditional estimation")

    return ScoringRuleApproximator(
        adapter=adapter,
        inference_network=scoring_rule_inference_network_with_multiple_parametric_scores,
        summary_network=summary_network,
    )


@pytest.fixture(
    params=[
        "scoring_rule_approximator_with_single_parametric_score",
        "scoring_rule_approximator_with_multiple_parametric_scores",
    ]
)
def scoring_rule_approximator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "point_approximator_without_parametric_score",
        "scoring_rule_approximator_with_single_parametric_score",
        "scoring_rule_approximator_with_multiple_parametric_scores",
    ]
)
def scoring_rule_approximator_any(request):
    return request.getfixturevalue(request.param)
