import keras
from tests.utils import check_combination_simulator_adapter


def test_approximator_sample(continuous_approximator_ensemble, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    continuous_approximator_ensemble.build(batch_shapes)

    samples = continuous_approximator_ensemble.sample(num_samples=2, conditions=data)

    assert isinstance(samples, dict)

    for samples_value in samples.values():
        assert isinstance(samples_value, dict)
