import keras
import inspect
import textwrap
from functools import wraps

from keras.src import dtype_policies
from keras.src import tree
from keras.src.backend.common.name_scope import current_path
from keras.src.utils import python_utils
from keras import Operation
from keras.saving import get_registered_name, get_registered_object

from bayesflow.utils.serialization import serialize, deserialize


class BayesFlowSerializableDict:
    def __init__(self, **config):
        self.config = config

    def serialize(self):
        return serialize(self.config)


class BaseLayer(keras.Layer):
    def __new__(cls, *args, **kwargs):
        """We override __new__ to saving serializable constructor arguments.

        These arguments are used to auto-generate an object serialization
        config, which enables user-created subclasses to be serializable
        out of the box in most cases without forcing the user
        to manually implement `get_config()`.
        """

        # Adapted from keras.Operation.__new__, to support all serializable objects, instead
        # of only basic types.

        instance = super(Operation, cls).__new__(cls)

        # Generate a config to be returned by default by `get_config()`.
        arg_names = inspect.getfullargspec(cls.__init__).args
        kwargs.update(dict(zip(arg_names[1 : len(args) + 1], args)))

        # Explicitly serialize `dtype` to support auto_config
        dtype = kwargs.get("dtype", None)
        if dtype is not None and isinstance(dtype, dtype_policies.DTypePolicy):
            # For backward compatibility, we use a str (`name`) for
            # `DTypePolicy`
            if dtype.quantization_mode is None:
                kwargs["dtype"] = dtype.name
            # Otherwise, use `dtype_policies.serialize`
            else:
                kwargs["dtype"] = dtype_policies.serialize(dtype)

        # Adaptation: we allow all registered serializable objects
        supported_types = (str, int, float, bool, type(None))
        try:
            flat_arg_values = tree.flatten(kwargs)
            auto_config = True
            for value in flat_arg_values:
                is_serializable = get_registered_object(get_registered_name(type(value))) is not None
                is_class = inspect.isclass(value)
                if not (isinstance(value, supported_types) or is_serializable or is_class):
                    auto_config = False
                    break
        except TypeError:
            auto_config = False
        try:
            instance._lock = False
            if auto_config:
                instance._auto_config = BayesFlowSerializableDict(**kwargs)
            else:
                instance._auto_config = None
            instance._lock = True
        except RecursionError:
            # Setting an instance attribute in __new__ has the potential
            # to trigger an infinite recursion if a subclass overrides
            # setattr in an unsafe way.
            pass

        ### from keras.Layer.__new__

        # Wrap the user-provided `build` method in the `build_wrapper`
        # to add name scope support and serialization support.
        original_build_method = instance.build

        @wraps(original_build_method)
        def build_wrapper(*args, **kwargs):
            with instance._open_name_scope():
                instance._path = current_path()
                original_build_method(*args, **kwargs)
            # Record build config.
            signature = inspect.signature(original_build_method)
            instance._build_shapes_dict = signature.bind(*args, **kwargs).arguments
            # Set built, post build actions, and lock state.
            instance.built = True
            instance._post_build()
            instance._lock_state()

        instance.build = build_wrapper

        # Wrap the user-provided `quantize` method in the `quantize_wrapper`
        # to add tracker support.
        original_quantize_method = instance.quantize

        @wraps(original_quantize_method)
        def quantize_wrapper(mode, **kwargs):
            instance._check_quantize_args(mode, instance.compute_dtype)
            instance._tracker.unlock()
            try:
                original_quantize_method(mode, **kwargs)
            except Exception:
                raise
            finally:
                instance._tracker.lock()

        instance.quantize = quantize_wrapper

        return instance

    @python_utils.default
    def get_config(self):
        """Returns the config of the object.

        An object config is a Python dictionary (serializable)
        containing the information needed to re-instantiate it.
        """

        # Adapted from Operations.get_config to support specifying a default configuration in
        # subclasses, without giving up on the automatic config functionality.
        config = super().get_config()
        if not python_utils.is_default(self.get_config):
            # In this case the subclass implements get_config()
            return config

        # In this case the subclass doesn't implement get_config():
        # Let's see if we can autogenerate it.
        if getattr(self, "_auto_config", None) is not None:
            xtra_args = set(config.keys())
            config.update(self._auto_config.config)
            # Remove args non explicitly supported
            argspec = inspect.getfullargspec(self.__init__)
            if argspec.varkw != "kwargs":
                for key in xtra_args - xtra_args.intersection(argspec.args[1:]):
                    config.pop(key, None)
            return config
        else:
            raise NotImplementedError(
                textwrap.dedent(
                    f"""
        Object {self.__class__.__name__} was created by passing
        non-serializable argument values in `__init__()`,
        and therefore the object must override `get_config()` in
        order to be serializable. Please implement `get_config()`.

        Example:

        class CustomLayer(keras.layers.Layer):
            def __init__(self, arg1, arg2, **kwargs):
                super().__init__(**kwargs)
                self.arg1 = arg1
                self.arg2 = arg2

            def get_config(self):
                config = super().get_config()
                config.update({{
                    "arg1": self.arg1,
                    "arg2": self.arg2,
                }})
                return config"""
                )
            )

    @classmethod
    def from_config(cls, config):
        """Creates an operation from its config.

        This method is the reverse of `get_config`, capable of instantiating the
        same operation from the config dictionary.

        Note: If you override this method, you might receive a serialized dtype
        config, which is a `dict`. You can deserialize it as follows:

        ```python
        if "dtype" in config and isinstance(config["dtype"], dict):
            policy = dtype_policies.deserialize(config["dtype"])
        ```

        Args:
            config: A Python dictionary, typically the output of `get_config`.

        Returns:
            An operation instance.
        """
        # Adapted from keras.Operation.from_config to use our deserialize function
        # Explicitly deserialize dtype config if needed. This enables users to
        # directly interact with the instance of `DTypePolicy`.
        if "dtype" in config and isinstance(config["dtype"], dict):
            config = config.copy()
            policy = dtype_policies.deserialize(config["dtype"])
            if not isinstance(policy, dtype_policies.DTypePolicyMap) and policy.quantization_mode is None:
                # For backward compatibility, we use a str (`name`) for
                # `DTypePolicy`
                policy = policy.name
            config["dtype"] = policy
        try:
            return cls(**deserialize(config))
        except Exception as e:
            raise TypeError(
                f"Error when deserializing class '{cls.__name__}' using config={config}.\n\nException encountered: {e}"
            )
