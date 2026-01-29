import numpy as np
from bayesflow.scores import ParametricDistributionScore
from tests.utils import check_combination_simulator_adapter


def test_approximator_log_prob(point_approximator, simulator, batch_size, num_samples, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    data = simulator.sample((batch_size,))

    batch = adapter(data)
    point_approximator.build_from_data(batch)

    log_prob = point_approximator.log_prob(data=data)
    parametric_scores = [
        score
        for score in point_approximator.inference_network.scores.values()
        if isinstance(score, ParametricDistributionScore)
    ]

    if len(parametric_scores) > 1:
        assert isinstance(log_prob, dict)
        for score_key, score_log_prob in log_prob.items():
            assert isinstance(score_log_prob, np.ndarray)
            assert score_log_prob.shape == (batch_size,)

    # If only one score is available, the outer nesting should be dropped.
    else:
        assert isinstance(log_prob, np.ndarray)
        assert log_prob.shape == (batch_size,)
