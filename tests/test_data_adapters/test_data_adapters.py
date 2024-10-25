from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)
import numpy as np


def test_cycle_consistency(data_adapter, random_data):
    processed = data_adapter(random_data)
    deprocessed = data_adapter(processed, inverse=True)

    for key, value in random_data.items():
        assert key in deprocessed
        assert np.allclose(value, deprocessed[key])


def test_serialize_deserialize(data_adapter, custom_objects):
    serialized = serialize(data_adapter)
    deserialized = deserialize(serialized, custom_objects)
    reserialized = serialize(deserialized)

    assert reserialized.keys() == serialized.keys()
    for key in reserialized:
        assert reserialized[key] == serialized[key]
