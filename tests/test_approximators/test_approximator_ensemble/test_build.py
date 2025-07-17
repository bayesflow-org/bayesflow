import keras
from tests.utils import check_combination_simulator_adapter


def test_build_continuous(continuous_approximator_ensemble, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    print(batch_shapes)
    continuous_approximator_ensemble.build(batch_shapes)

    for member in continuous_approximator_ensemble.approximators.values():
        for layer in member.standardize_layers.values():
            assert layer.built
            for count in layer.count:
                assert count == 0.0


def test_build_model_comparison(
    model_comparison_approximator_ensemble, model_comparison_simulator, batch_size, model_comparison_adapter
):
    check_combination_simulator_adapter(model_comparison_simulator, model_comparison_adapter)

    num_batches = 4
    data = model_comparison_simulator.sample((num_batches * batch_size,))

    batch = model_comparison_adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    model_comparison_approximator_ensemble.build(batch_shapes)

    for member in model_comparison_approximator_ensemble.approximators.values():
        for layer in member.standardize_layers.values():
            assert layer.built
            for count in layer.count:
                assert count == 0.0
