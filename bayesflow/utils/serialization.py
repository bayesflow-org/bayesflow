import keras


PREFIX = "_bayesflow_"


def serialize_value_or_type(config, name, obj):
    """Serialize an object that can be either a value or a type
    and add it to a copy of the supplied dictionary.

    Parameters
    ----------
    config  : dict
        Dictionary to add the serialized object to. This function does not
        modify the dictionary in place, but returns a modified copy.
    name    : str
        Name of the obj that should be stored. Required for later deserialization.
    obj     : object or type
        The object to serialize. If `obj` is of type `type`, we use
        `keras.saving.get_registered_name` to obtain the registered type name.
        If it is not a type, we try to serialize it as a Keras object.

    Returns
    -------
    updated_config  : dict
        Updated dictionary with a new key `"_bayesflow_<name>_type"` or
        `"_bayesflow_<name>_val"`. The prefix is used to avoid name collisions,
        the suffix indicates how the stored value has to be deserialized.

    Notes
    -----
    We allow strings or `type` parameters at several places to instantiate objects
    of a given type (e.g., `subnet` in `CouplingFlow`). As `type` objects cannot
    be serialized, we have to distinguish the two cases for serialization and
    deserialization. This function is a helper function to standardize and
    simplify this.
    """
    updated_config = config.copy()
    if isinstance(obj, type):
        updated_config[f"{PREFIX}{name}_type"] = keras.saving.get_registered_name(obj)
    else:
        updated_config[f"{PREFIX}{name}_val"] = keras.saving.serialize_keras_object(obj)
    return updated_config


def deserialize_value_or_type(config, name):
    """Deserialize an object that can be either a value or a type and add
    it to the supplied dictionary.

    Parameters
    ----------
    config  : dict
        Dictionary containing the object to deserialize. If a type was
        serialized, it should contain the key `"_bayesflow_<name>_type"`.
        If an object was serialized, it should contain the key
        `"_bayesflow_<name>_val"`. In a copy of this dictionary,
        the item will be replaced with the key `name`.
    name    : str
        Name of the object to deserialize.

    Returns
    -------
    updated_config  : dict
        Updated dictionary with a new key `name`, with a value that is either
        a type or an object.

    See Also
    --------
    serialize_value_or_type
    """
    updated_config = config.copy()
    if f"{PREFIX}{name}_type" in config:
        updated_config[name] = keras.saving.get_registered_object(config[f"{PREFIX}{name}_type"])
        del updated_config[f"{PREFIX}{name}_type"]
    elif f"{PREFIX}{name}_val" in config:
        updated_config[name] = keras.saving.deserialize_keras_object(config[f"{PREFIX}{name}_val"])
        del updated_config[f"{PREFIX}{name}_val"]
    return updated_config
