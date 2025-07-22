from keras.saving import deserialize_keras_object, serialize_keras_object
import pickle
from pathlib import Path


def dump_path(object, filepath: Path | str):
    with open(filepath, "wb") as f:
        pickle.dump(object, f)


def load_path(filepath: Path | str):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def save_config(object, filepath: Path | str):
    dump_path(serialize_keras_object(object), filepath)


def load_from_config(filepath: Path | str, custom_objects=None):
    config = load_path(filepath)
    return deserialize_keras_object(config, custom_objects=custom_objects)
