import keras
import numpy as np
from bayesflow.scores import ParametricDistributionScore


def test_approximator_sample(point_approximator, simulator, batch_size, num_samples, adapter):
    data = simulator.sample((batch_size,))

    batch = adapter(data)
    point_approximator.build_from_data(batch)

    samples = point_approximator.sample(num_samples=num_samples, conditions=data)

    assert isinstance(samples, dict)

    print(keras.tree.map_structure(keras.ops.shape, samples))

    # Expect doubly nested sample dictionary if more than one samplable score is available.
    scores_for_sampling = [
        score
        for score in point_approximator.inference_network.scores.values()
        if isinstance(score, ParametricDistributionScore)
    ]

    if len(scores_for_sampling) > 1:
        for score_key, score_samples in samples.items():
            for variable, variable_estimates in score_samples.items():
                assert isinstance(variable_estimates, np.ndarray)
                assert variable_estimates.shape[:-1] == (batch_size, num_samples)

    # If only one score is available, the outer nesting should be dropped.
    else:
        for variable, variable_estimates in samples.items():
            assert isinstance(variable_estimates, np.ndarray)
            assert variable_estimates.shape[:-1] == (batch_size, num_samples)
