from collections.abc import Collection
from copy import copy

import inspect
import keras

from bayesflow.utils import filter_keys
from bayesflow.utils.inspect_utils import get_calling_frame_info


def deserialize(obj, custom_objects=None, module_objects=None):
    if inspect.isclass(obj):
        return keras.saving.get_registered_object(obj, custom_objects=custom_objects, module_objects=module_objects)
    return keras.saving.deserialize_keras_object(obj, custom_objects=custom_objects, module_objects=module_objects)


class Serializable:
    def __init_subclass__(cls, **kwargs):
        # get the calling module's name, e.g. "bayesflow.networks.inference_network"
        stack = inspect.stack()
        module = inspect.getmodule(stack[1][0])
        package = copy(module.__name__)

        name = copy(cls.__name__)

        # register subclasses as keras serializable
        keras.saving.register_keras_serializable(package=package, name=name)(cls)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        kwargs = config.get("constructor_arguments", {})
        kwargs = {key: deserialize(value, custom_objects) for key, value in kwargs.items()}

        instance = cls(**kwargs)

        stateful_fields = config.get("stateful_fields", {})
        for field_name, field_value in stateful_fields.items():
            setattr(instance, field_name, deserialize(field_value, custom_objects))

        # TODO: what do we do with the keras config values? e.g. "name", "trainable", etc.

        return instance

    def get_config(self):
        base_config = super().get_config() if hasattr(super(), "get_config") else {}

        config = getattr(self, "config", {})

        config["stateful_fields"] = {key: serialize(getattr(self, key)) for key in config.get("stateful_fields", [])}

        return base_config | config

    def initialize_config(self, *, include: list[str] = None, exclude: list[str] = None, stateful: list[str] = None):
        """
        Initialize the configuration dictionary for a Serializable subclass by looking up constructor arguments and
        remembering stateful fields to be serialized later.

        Call this method after calling super().__init__() in the constructor of a subclass of Serializable.

        Parameters
        ----------
        include : list[str], optional
            List of keys to include in the configuration. If None, no keys are included based on this list.
        exclude : list[str], optional
            List of keys to exclude from the configuration. If None, no keys are excluded based on this list.
        stateful : list[str], optional
            List of stateful fields to include in the configuration. If None, no stateful fields are included.

        Returns
        -------
        dict
            The configuration dictionary containing constructor arguments and stateful fields.

        Raises
        ------
        RuntimeError
            If this method is called from a non-constructor context.
        """
        if stateful is None:
            stateful = []

        # ensure that the frame above the current one is a constructor
        frame_info = get_calling_frame_info()

        # check that the calling function is an __init__()
        if frame_info.function != "__init__":
            raise RuntimeError("Cannot automatically initialize config from non-constructor context.")

        # get argument info from the calling frame
        arginfo = inspect.getargvalues(frame_info.frame)

        # drop self from locals and turn args into key-value dictionary
        args = {key: arginfo.locals[key] for key in arginfo.args[1:]}

        args = filter_keys(args, include=include, exclude=exclude, strict=True)

        self.config = {
            "constructor_arguments": serialize(args),
            "stateful_fields": list(stateful),
        }

        return self.config


def serialize(obj):
    if not isinstance(obj, str) and isinstance(obj, Collection):
        return keras.tree.map_structure(serialize, obj)

    if inspect.isclass(obj):
        return keras.saving.get_registered_name(obj)
    return keras.saving.serialize_keras_object(obj)
