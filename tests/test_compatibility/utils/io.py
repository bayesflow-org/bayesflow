import inspect
from keras.saving import deserialize_keras_object, serialize_keras_object
from pathlib import Path
import pickle
import re


def get_path(data_dir: Path | str = "", filename: str = "", *, create: bool = False) -> Path:
    frame = inspect.stack()[1]
    base_path = Path(inspect.stack()[1].filename[:-3])
    function_name = frame.function
    if "self" in frame[0].f_locals:
        filepath = base_path / frame[0].f_locals["self"].__class__.__name__ / function_name
    else:
        filepath = base_path / function_name
    filepath = Path(data_dir) / filepath.relative_to(Path("tests").absolute())
    if create is True:
        filepath.mkdir(parents=True, exist_ok=True)
    if filename:
        return filepath / filename
    return filepath


def get_valid_filename(name):
    s = str(name).strip().replace(" ", "_")
    s = re.sub(r"(?u)[^-\w.]", "_", s)
    if s in {"", ".", ".."}:
        raise ValueError("Could not derive file name from '%s'" % name)
    return s


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
