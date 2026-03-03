import pytest
import numpy as np
from bayesflow.scoring_rules import ParametricDistributionScore
from tests.utils import check_combination_simulator_adapter


def test_log_prob(scoring_rule_approximator_any, simulator, batch_size, num_samples, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    if not scoring_rule_approximator_any.has_distribution:
        pytest.skip("No parametric distribution scores available")

    data = simulator.sample(batch_size)

    batch = adapter(data)
    scoring_rule_approximator_any.build_from_data(batch)

    parametric_scores = {
        key: score
        for key, score in scoring_rule_approximator_any.inference_network.scoring_rules.items()
        if isinstance(score, ParametricDistributionScore)
    }

    # merge_scores=True (default): always returns a single merged array
    log_prob_merged = scoring_rule_approximator_any.log_prob(data=data)
    assert isinstance(log_prob_merged, np.ndarray)
    assert log_prob_merged.shape == (batch_size,)

    # merge_scores=False: returns a dict when multiple scores, plain array when single
    log_prob_separate = scoring_rule_approximator_any.log_prob(data=data, merge_scores=False)
    if len(parametric_scores) > 1:
        assert isinstance(log_prob_separate, dict)
        for score_key, score_log_prob in log_prob_separate.items():
            assert isinstance(score_log_prob, np.ndarray)
            assert score_log_prob.shape == (batch_size,)
    else:
        assert isinstance(log_prob_separate, np.ndarray)
        assert log_prob_separate.shape == (batch_size,)
