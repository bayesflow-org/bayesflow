from bayesflow.utils.serialization import serialize, deserialize
import keras


def test_serialize_deserialize(metric, random_samples):
    metric.update_state(keras.random.normal((2, 3)), keras.random.normal((2, 3)))

    serialized = serialize(metric)
    deserialized = deserialize(serialized)
    reserialized = serialize(deserialized)

    assert reserialized == serialized
