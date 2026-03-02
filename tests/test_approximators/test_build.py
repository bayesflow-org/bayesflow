import keras
from tests.utils import check_combination_simulator_adapter


def test_build(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    def ensure_standardize_layers_built(approximator):
        if hasattr(approximator, "standardize_layers"):
            for layer in approximator.standardize_layers.values():
                assert layer.built
                for count in layer.count:
                    assert count == 0.0
        elif hasattr(approximator, "approximators"):  # approximator is EnsembleApproximator
            for wrapped_approx in approximator.approximators.values():
                ensure_standardize_layers_built(wrapped_approx)
        else:
            raise ValueError(
                "Approximator needs to have one of the following attributes: standardize_layers, approximators"
            )

    ensure_standardize_layers_built(approximator)
