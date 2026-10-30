import keras
from tests.utils import check_combination_simulator_adapter


def test_build(estimator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    estimator.build(batch_shapes)

    def ensure_standardize_layers_built(estimator):
        if hasattr(estimator, "standardizer"):
            for layer in estimator.standardize_layers.values():
                assert layer.built
                for count in layer.count:
                    assert count == 0.0
        elif hasattr(estimator, "estimators"):  # estimator is EnsembleEstimator
            for wrapped_estimator in estimator.estimators.values():
                ensure_standardize_layers_built(wrapped_estimator)
        else:
            raise ValueError("Estimator needs to have one of the following attributes: standardize_layers, estimators")

    ensure_standardize_layers_built(estimator)
