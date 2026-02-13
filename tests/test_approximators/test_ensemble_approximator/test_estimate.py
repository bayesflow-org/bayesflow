import keras
from tests.utils import check_combination_simulator_adapter


def test_approximator_estimate_separate(ensemble_approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    ensemble_approximator.build(batch_shapes)

    estimates = ensemble_approximator.estimate_separate(data)

    assert isinstance(estimates, dict)

    for estimates_value in estimates.values():
        assert isinstance(estimates_value, dict)
