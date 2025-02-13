from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)
import numpy as np


def test_cycle_consistency(adapter, random_data):
    processed = adapter(random_data)
    deprocessed = adapter(processed, inverse=True)

    for key, value in random_data.items():
        if key in ["d1", "d2"]:
            # dropped
            continue
        assert key in deprocessed
        assert np.allclose(value, deprocessed[key])


def test_serialize_deserialize(adapter, custom_objects, random_data):
    processed = adapter(random_data)
    serialized = serialize(adapter)
    deserialized = deserialize(serialized, custom_objects)
    reserialized = serialize(deserialized)

    assert reserialized.keys() == serialized.keys()
    for key in reserialized:
        assert reserialized[key] == serialized[key]

    random_data["foo"] = random_data["x1"]
    deserialized_processed = deserialized(random_data)
    for key, value in processed.items():
        assert np.allclose(value, deserialized_processed[key])
