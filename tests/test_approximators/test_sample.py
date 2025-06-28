import numpy as np
import keras
from tests.utils import check_combination_simulator_adapter


def test_approximator_sample(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    samples = approximator.sample(num_samples=2, conditions=data)

    assert isinstance(samples, dict)


def test_approximator_sample_keep_conditions(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)

    num_samples = 2
    samples_and_conditions = approximator.sample(num_samples=num_samples, conditions=data, keep_conditions=True)

    assert isinstance(samples_and_conditions, dict)

    # remove inference_variables from sample output and apply adapter
    inference_variables_keys = approximator.sample(num_samples=num_samples, conditions=data).keys()
    for key in inference_variables_keys:
        samples_and_conditions.pop(key)
    adapted_conditions = adapter(samples_and_conditions, strict=False)

    assert any(k in adapted_conditions for k in approximator.CONDITION_KEYS), (
        f"adapter(approximator.sample(..., keep_conditions=True)) must return at least one of"
        f"{approximator.CONDITION_KEYS!r}. Keys are {adapted_conditions.keys()}."
    )

    for key, value in adapted_conditions.items():
        assert value.shape[:2] == (num_batches * batch_size, num_samples), (
            f"{key} should have shape ({num_batches * batch_size}, {num_samples}, ...) but has {value.shape}."
        )

        if key in approximator.CONDITION_KEYS:
            assert np.all(np.ptp(value, axis=1) == 0), "Not all values are the same along axis 1"
