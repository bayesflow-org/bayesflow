import pytest
import keras
import numpy as np
from bayesflow.scoring_rules import ParametricDistributionScore
from tests.utils import check_combination_simulator_adapter


def test_sample(point_approximator, simulator, batch_size, num_samples, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    if not point_approximator.has_distribution:
        pytest.skip("No parametric distribution scores available")

    data = simulator.sample((batch_size,))

    batch = adapter(data)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    point_approximator.build(batch_shapes)

    scores_for_sampling = {
        key: score
        for key, score in point_approximator.inference_network.scoring_rules.items()
        if isinstance(score, ParametricDistributionScore)
    }

    # merge_scores=True (default): returns {variable: array} regardless of score count
    samples_merged = point_approximator.sample(num_samples=num_samples, conditions=data)
    assert isinstance(samples_merged, dict)
    for variable, variable_estimates in samples_merged.items():
        assert isinstance(variable_estimates, np.ndarray)
        assert variable_estimates.shape[:-1] == (batch_size, num_samples)

    # merge_scores=False: returns nested {score: {variable: array}} when multiple scores,
    # flat {variable: array} when single score
    samples_separate = point_approximator.sample(num_samples=num_samples, conditions=data, merge_scores=False)
    assert isinstance(samples_separate, dict)

    if len(scores_for_sampling) > 1:
        for score_key, score_samples in samples_separate.items():
            assert isinstance(score_samples, dict)
            for variable, variable_estimates in score_samples.items():
                assert isinstance(variable_estimates, np.ndarray)
                assert variable_estimates.shape[:-1] == (batch_size, num_samples)
    else:
        for variable, variable_estimates in samples_separate.items():
            assert isinstance(variable_estimates, np.ndarray)
            assert variable_estimates.shape[:-1] == (batch_size, num_samples)
