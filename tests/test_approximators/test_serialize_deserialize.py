import keras
import numpy as np

from ..utils import assert_models_equal


def test_serialize_deserialize_continuous_approximator(tmp_path, continuous_approximator):
    sample_data = {
        "mean": np.zeros((32, 10, 2)),
        "std": np.ones((32, 10, 1)),
        "x": np.random.standard_normal((32, 10, 2)),
    }

    sample_data = continuous_approximator.adapter(sample_data)

    continuous_approximator.build_from_data(sample_data)

    keras.saving.save_model(continuous_approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")
    assert_models_equal(continuous_approximator, loaded)

    # serialized = serialize(continuous_approximator)
    # deserialized = deserialize(serialized)
    # reserialized = serialize(deserialized)
    #
    # assert serialized == reserialized
