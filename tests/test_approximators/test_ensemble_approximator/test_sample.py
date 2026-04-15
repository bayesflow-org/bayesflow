import pytest
import keras
import numpy as np
from tests.utils import check_combination_simulator_adapter


def test_approximator_sample(ensemble_approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    ensemble_approximator.build(batch_shapes)

    samples = ensemble_approximator.sample(num_samples=2, conditions=data)

    assert isinstance(samples, dict)

    for samples_value in samples.values():
        assert isinstance(samples_value, np.ndarray)

    samples_seed42_1 = ensemble_approximator.sample(num_samples=2, conditions=data, seed=42)
    samples_seed42_2 = ensemble_approximator.sample(num_samples=2, conditions=data, seed=42)

    for key in samples.keys():
        np.testing.assert_allclose(
            samples_seed42_1[key],
            samples_seed42_2[key],
            err_msg=f"{key}: samples differ for identical seed",
        )
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(
                samples_seed42_1[key],
                samples[key],
            )
