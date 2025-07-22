import keras
import numpy as np
from tests.utils import check_combination_simulator_adapter


# def test_predict(model_comparison_approximator_ensemble, model_comparison_train_dataset_for_ensemble, simulator):
#     approximator_ensemble = model_comparison_approximator_ensemble
#     data_shapes = keras.tree.map_structure(keras.ops.shape, model_comparison_train_dataset_for_ensemble[0])
#     approximator_ensemble.build(data_shapes)
#     approximator_ensemble.compute_metrics(**model_comparison_train_dataset_for_ensemble[0])
#
#     num_conditions = 2
#     conditions = simulator.sample(num_conditions)
#     predictions = approximator_ensemble.predict(conditions=conditions)
#
#     for predictions_value in predictions.values():
#         assert isinstance(predictions_value, np.ndarray)
#         assert predictions_value.shape[0] == num_conditions


def test_predict_model_comparison(
    model_comparison_approximator_ensemble, model_comparison_simulator, batch_size, model_comparison_adapter
):
    check_combination_simulator_adapter(model_comparison_simulator, model_comparison_adapter)

    num_batches = 4
    data = model_comparison_simulator.sample((num_batches * batch_size,))

    batch = model_comparison_adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    model_comparison_approximator_ensemble.build(batch_shapes)

    num_conditions = 2
    conditions = model_comparison_simulator.sample(num_conditions)
    print(conditions)
    predictions = model_comparison_approximator_ensemble.predict(conditions=conditions)

    for predictions_value in predictions.values():
        assert isinstance(predictions_value, np.ndarray)
        assert predictions_value.shape[0] == num_conditions
