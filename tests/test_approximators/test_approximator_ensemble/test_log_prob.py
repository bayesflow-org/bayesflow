import keras
import numpy as np
from tests.utils import check_combination_simulator_adapter


def test_approximator_log_prob(continuous_approximator_ensemble, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    continuous_approximator_ensemble.build(batch_shapes)

    log_prob = continuous_approximator_ensemble.log_prob(data=data)
    assert isinstance(log_prob, dict)

    for log_prob_value in log_prob.values():
        assert isinstance(log_prob_value, (np.ndarray, dict))
