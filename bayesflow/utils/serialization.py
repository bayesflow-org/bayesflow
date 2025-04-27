from copy import copy

import builtins
import inspect
import keras
import numpy as np
import sys
from warnings import warn

# this import needs to be exactly like this to work with monkey patching
from keras.saving import deserialize_keras_object

from .context_managers import monkey_patch
from .decorators import allow_args


PREFIX = "_bayesflow_"

_type_prefix = "__bayesflow_type__"


def serialize_value_or_type(config, name, obj):
    """This function is deprecated."""
    warn(
        "This method is deprecated. It was replaced by bayesflow.utils.serialization.serialize.",
        DeprecationWarning,
        stacklevel=2,
    )


def deserialize_value_or_type(config, name):
    """This function is deprecated."""
    warn(
        "This method is deprecated. It was replaced by bayesflow.utils.serialization.deserialize.",
        DeprecationWarning,
        stacklevel=2,
    )


def deserialize(config: dict, custom_objects=None, safe_mode=True, **kwargs):
    """Deserialize an object serialized with :py:func:`serialize`.

    Wrapper function around `keras.saving.deserialize_keras_object` to enable deserialization of
    classes.

    Parameters
    ----------
    config : dict
        Python dict describing the object.
    custom_objects : dict, optional
        Python dict containing a mapping between custom object names and the corresponding
        classes or functions. Forwarded to `keras.saving.deserialize_keras_object`.
    safe_mode : bool, optional
        Boolean, whether to disallow unsafe lambda deserialization. When safe_mode=False,
        loading an object has the potential to trigger arbitrary code execution. This argument
        is only applicable to the Keras v3 model format. Defaults to True.
        Forwarded to `keras.saving.deserialize_keras_object`.

    Returns
    -------
    obj :
        The object described by the config dictionary.

    Raises
    ------
    ValueError
        If a type in the config can not be deserialized.

    See Also
    --------
    serialize
    """
    with monkey_patch(deserialize_keras_object, deserialize) as original_deserialize:
        if isinstance(config, str) and config.startswith(_type_prefix):
            # we marked this as a type during serialization
            config = config[len(_type_prefix) :]
            tp = keras.saving.get_registered_object(
                # TODO: can we pass module objects without overwriting numpy's dict with builtins?
                config,
                custom_objects=custom_objects,
                module_objects=np.__dict__ | builtins.__dict__,
            )
            if tp is None:
                raise ValueError(
                    f"Could not deserialize type {config!r}. Make sure it is registered with "
                    f"`keras.saving.register_keras_serializable` or pass it in `custom_objects`."
                )
            return tp
        if inspect.isclass(config):
            # add this base case since keras does not cover it
            return config

        obj = original_deserialize(config, custom_objects=custom_objects, safe_mode=safe_mode, **kwargs)

        return obj


@allow_args
def serializable(cls, package: str | None = None, name: str | None = None):
    """Register class as Keras serialize.

    Wrapper function around `keras.saving.register_keras_serializable` to automatically
    set the `package` and `name` arguments.

    Parameters
    ----------
    cls : type
        The class to register.
    package : str, optional
        `package` argument forwarded to `keras.saving.register_keras_serializable`.
        If None is provided, the package is automatically inferred using the __name__
        attribute of the module the class resides in.
    name : str, optional
        `name` argument forwarded to `keras.saving.register_keras_serializable`.
        If None is provided, the classe's __name__ attribute is used.
    """
    if package is None:
        frame = sys._getframe(2)
        g = frame.f_globals
        package = g.get("__name__", "bayesflow")

    if name is None:
        name = copy(cls.__name__)

    # register subclasses as keras serializable
    return keras.saving.register_keras_serializable(package=package, name=name)(cls)


def serialize(obj):
    """Serialize an object using Keras.

    Wrapper function around `keras.saving.serialize_keras_object`, which adds the
    ability to serialize classes.

    Parameters
    ----------
    object : Keras serializable object, or class
        The object to serialize

    Returns
    -------
    config : dict
        A python dict that represents the object. The python dict can be deserialized via
        :py:func:`deserialize`.

    See Also
    --------
    deserialize
    """
    if isinstance(obj, (tuple, list, dict)):
        return keras.tree.map_structure(serialize, obj)
    elif inspect.isclass(obj):
        return _type_prefix + keras.saving.get_registered_name(obj)

    return keras.saving.serialize_keras_object(obj)
