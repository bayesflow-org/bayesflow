import keras
from tests.utils import check_combination_simulator_adapter


def test_approximator_estimate(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    estimates = approximator.estimate(data)

    assert isinstance(estimates, dict)
    print(keras.tree.map_structure(keras.ops.shape, estimates))
