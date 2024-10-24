from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)
import numpy as np


def test_cycle_consistency(adapter, random_data):
    processed = adapter(random_data)
    deprocessed = adapter(processed, inverse=True)

    for key, value in random_data.items():
        assert key in deprocessed
        assert np.allclose(value, deprocessed[key])


def test_serialize_deserialize(adapter, custom_objects):
    serialized = serialize(adapter)
    deserialized = deserialize(serialized, custom_objects)
    reserialized = serialize(deserialized)

    assert reserialized.keys() == serialized.keys()
    for key in reserialized:
        assert reserialized[key] == serialized[key]
