from copy import deepcopy
import keras


def normalize_dtype(config):
    """Convert dtypes with DTypePolicy to simple strings"""
    config = deepcopy(config)

    def walk_dictionary(cur_dict):
        # walks the dicitonary and modifies entries in-place
        for key, value in cur_dict.items():
            if key == "dtype" and isinstance(value, dict):
                if value.get("class_name", "") == "DTypePolicy":
                    cur_dict[key] = value["config"]["name"]
                continue
        if isinstance(value, dict):
            walk_dictionary(value)

    walk_dictionary(config)
    return config


def normalize_config(config):
    config = normalize_dtype(config)
    config = keras.tree.lists_to_tuples(config)
