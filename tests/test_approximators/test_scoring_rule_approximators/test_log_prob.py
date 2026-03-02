import numpy as np
from bayesflow.scoring_rules import ParametricDistributionScore
from tests.utils import check_combination_simulator_adapter


def test_approximator_log_prob(
    scoring_rule_approximator_with_multiple_parametric_scores, simulator, batch_size, adapter
):
    approx = scoring_rule_approximator_with_multiple_parametric_scores
    check_combination_simulator_adapter(simulator, adapter)

    data = simulator.sample((batch_size,))

    batch = adapter(data)
    approx.build_from_data(batch)

    log_prob = approx.log_prob(data=data, merge_scores=False)
    parametric_scoring_rules = [
        score
        for score in approx.inference_network.scoring_rules.values()
        if isinstance(score, ParametricDistributionScore)
    ]

    if len(parametric_scoring_rules) > 1:
        assert isinstance(log_prob, dict)
        for score_key, score_log_prob in log_prob.items():
            assert isinstance(score_log_prob, np.ndarray)
            assert score_log_prob.shape == (batch_size,)

    # If only one score is available, the outer nesting should be dropped.
    else:
        assert isinstance(log_prob, np.ndarray)
        assert log_prob.shape == (batch_size,)
