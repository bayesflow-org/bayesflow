import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    serialize_keras_object as serialize,
)


def test_cycle_consistency(data_adapter, random_data):
    processed = data_adapter.configure(random_data)
    deprocessed = data_adapter.deconfigure(processed)

    for key, value in random_data.items():
        assert key in deprocessed
        assert keras.ops.all(keras.ops.isclose(value, deprocessed[key]))


def test_serialize_deserialize(data_adapter, custom_objects):
    serialized = serialize(data_adapter)
    deserialized = deserialize(serialized, custom_objects)
    reserialized = serialize(deserialized)

    assert reserialized == serialized
