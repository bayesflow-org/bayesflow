import numpy as np
from bayesflow.scores import ParametricDistributionScore


def test_log_prob(point_approximator, simulator, batch_size, num_samples, adapter):
    data = simulator.sample((batch_size,))

    batch = adapter(data)
    point_approximator.build_from_data(batch)

    parametric_scores = {
        key: score
        for key, score in point_approximator.inference_network.scores.items()
        if isinstance(score, ParametricDistributionScore)
    }

    # merge_scores=True (default): always returns a single merged array
    log_prob_merged = point_approximator.log_prob(data=data)
    assert isinstance(log_prob_merged, np.ndarray)
    assert log_prob_merged.shape == (batch_size,)

    # merge_scores=False: returns a dict when multiple scores, plain array when single
    log_prob_separate = point_approximator.log_prob(data=data, merge_scores=False)
    if len(parametric_scores) > 1:
        assert isinstance(log_prob_separate, dict)
        for score_key, score_log_prob in log_prob_separate.items():
            assert isinstance(score_log_prob, np.ndarray)
            assert score_log_prob.shape == (batch_size,)
    else:
        assert isinstance(log_prob_separate, np.ndarray)
        assert log_prob_separate.shape == (batch_size,)
