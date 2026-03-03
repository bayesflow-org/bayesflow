import keras
import numpy as np
from bayesflow.scoring_rules import ParametricDistributionScore
from tests.utils import check_combination_simulator_adapter


def test_approximator_sample(
    scoring_rule_approximator_with_multiple_parametric_scores, simulator, batch_size, num_samples, adapter
):
    approx = scoring_rule_approximator_with_multiple_parametric_scores
    check_combination_simulator_adapter(simulator, adapter)

    data = simulator.sample((batch_size,))

    batch = adapter(data)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approx.build(batch_shapes)

    samples = approx.sample(num_samples=num_samples, conditions=data, merge_scores=False)

    assert isinstance(samples, dict)

    # Expect doubly nested sample dictionary if more than one samplable score is available.
    scoring_rules_for_sampling = [
        score
        for score in approx.inference_network.scoring_rules.values()
        if isinstance(score, ParametricDistributionScore)
    ]

    if len(scoring_rules_for_sampling) > 1:
        for score_key, score_samples in samples.items():
            for variable, variable_estimates in score_samples.items():
                assert isinstance(variable_estimates, np.ndarray)
                assert variable_estimates.shape[:-1] == (batch_size, num_samples)

    # If only one score is available, the outer nesting should be dropped.
    else:
        for variable, variable_estimates in samples.items():
            assert isinstance(variable_estimates, np.ndarray)
            assert variable_estimates.shape[:-1] == (batch_size, num_samples)
