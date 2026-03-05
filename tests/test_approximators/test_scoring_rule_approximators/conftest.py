import pytest


@pytest.fixture()
def scoring_rule_approximator_with_multiple_parametric_scores(adapter, summary_network):
    from bayesflow import ScoringRuleApproximator
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import MvNormalScore

    if "-> 'inference_conditions'" not in str(adapter) and "-> 'summary_conditions'" not in str(adapter):
        pytest.skip("Scoring rule approximator does not support unconditional estimation")

    return ScoringRuleApproximator(
        adapter=adapter,
        inference_network=ScoringRuleNetwork(
            scoring_rules=dict(
                mvn1=MvNormalScore(),
                mvn2=MvNormalScore(),
            ),
        ),
        summary_network=summary_network,
    )
