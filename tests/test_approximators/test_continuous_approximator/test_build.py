import keras
from tests.utils import check_combination_simulator_adapter


def test_build(continuous_approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    continuous_approximator.build(batch_shapes)
    for layer in continuous_approximator.standardizer.standardize_layers.values():
        assert layer.built
        for count in layer.count:
            assert count == 0.0
