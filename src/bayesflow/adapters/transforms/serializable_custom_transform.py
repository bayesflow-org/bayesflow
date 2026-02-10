from collections.abc import Callable
import numpy as np
from keras.saving import (
    get_registered_name,
    get_registered_object,
)

from bayesflow.utils.serialization import deserialize, serializable, serialize
from .elementwise_transform import ElementwiseTransform
from ...utils import filter_kwargs
import inspect


@serializable("bayesflow.adapters")
class SerializableCustomTransform(ElementwiseTransform):
    """
    Transforms a parameter using a pair of registered serializable forward and inverse functions.

    Parameters
    ----------
    forward : function, no lambda
        Registered serializable function to transform the data in the forward pass.
        For the adapter to be serializable, this function has to be serializable
        as well (see Notes). Therefore, only proper functions and no lambda
        functions can be used here.
    inverse : function, no lambda
        Function to transform the data in the inverse pass.
        For the adapter to be serializable, this function has to be serializable
        as well (see Notes). Therefore, only proper functions and no lambda
        functions can be used here.

    Raises
    ------
    ValueError
        When the provided functions are not registered serializable functions.

    Notes
    -----
    Important: The forward and inverse functions have to be registered with Keras.
    To do so, use the `@keras.saving.register_keras_serializable` decorator.
    They must also be registered (and identical) when loading the adapter
    at a later point in time.

    """

    def __init__(
        self,
        *,
        forward: Callable[[np.ndarray, ...], np.ndarray],
        inverse: Callable[[np.ndarray, ...], np.ndarray],
    ):
        super().__init__()

        self._check_serializable(forward, label="forward")
        self._check_serializable(inverse, label="inverse")
        self._forward = forward
        self._inverse = inverse

    @classmethod
    def _check_serializable(cls, function, label=""):
        GENERAL_EXAMPLE_CODE = (
            "The example code below shows the structure of a correctly decorated function:\n\n"
            "```\n"
            "import keras\n\n"
            "@keras.saving.register_keras_serializable('custom')\n"
            f"def my_{label}(...):\n"
            "    [your code goes here...]\n"
            "```\n"
        )
        if function is None:
            raise TypeError(
                f"'{label}' must be a registered serializable function, was 'NoneType'.\n{GENERAL_EXAMPLE_CODE}"
            )
        registered_name = get_registered_name(function)
        # check if function is a lambda function
        if registered_name == "<lambda>":
            raise ValueError(
                f"The provided function for '{label}' is a lambda function, "
                "which cannot be serialized. "
                "Please provide a registered serializable function by using the "
                "@keras.saving.register_keras_serializable decorator."
                f"\n{GENERAL_EXAMPLE_CODE}"
            )
        if inspect.ismethod(function):
            raise ValueError(
                f"The provided value for '{label}' is a method, not a function. "
                "Methods cannot be serialized separately from their classes. "
                "Please provide a registered serializable function instead by "
                "moving the functionality to a function (i.e., outside of the class) and "
                "using the @keras.saving.register_keras_serializable decorator."
                f"\n{GENERAL_EXAMPLE_CODE}"
            )
        registered_object_for_name = get_registered_object(registered_name)
        if registered_object_for_name is None:
            try:
                source_max_lines = 5
                function_source_code = inspect.getsource(function).split("\n")
                if len(function_source_code) > source_max_lines:
                    function_source_code = function_source_code[:source_max_lines] + ["    [...]"]

                example_code = "For your provided function, this would look like this:\n\n"
                example_code += "\n".join(
                    ["```", "import keras\n", "@keras.saving.register_keras_serializable('custom')"]
                    + function_source_code
                    + ["```"]
                )
            except OSError:
                example_code = GENERAL_EXAMPLE_CODE
            raise ValueError(
                f"The provided function for '{label}' is not registered with Keras.\n"
                "Please register the function using the "
                "@keras.saving.register_keras_serializable decorator.\n"
                f"{example_code}"
            )
        if registered_object_for_name is not function:
            raise ValueError(
                f"The provided function for '{label}' does not match the function "
                f"registered under its name '{registered_name}'. "
                f"(registered function: {registered_object_for_name}, provided function: {function}). "
            )

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "SerializableCustomTransform":
        if get_registered_object(config["forward"]["config"], custom_objects) is None:
            provided_function_msg = ""
            if config["_forward_source_code"]:
                provided_function_msg = (
                    f"\nThe originally provided function was:\n\n```\n{config['_forward_source_code']}\n```"
                )
            raise TypeError(
                "\n\nPLEASE READ HERE:\n"
                "-----------------\n"
                "The forward function that was provided as `forward` "
                "is not registered with Keras, making deserialization impossible. "
                f"Please ensure that it is registered as '{config['forward']['config']}' and identical to the original "
                "function before loading your model."
                f"{provided_function_msg}"
            )
        if get_registered_object(config["inverse"]["config"], custom_objects) is None:
            provided_function_msg = ""
            if config["_inverse_source_code"]:
                provided_function_msg = (
                    f"\nThe originally provided function was:\n\n```\n{config['_inverse_source_code']}\n```"
                )
            raise TypeError(
                "\n\nPLEASE READ HERE:\n"
                "-----------------\n"
                "The inverse function that was provided as `inverse` "
                "is not registered with Keras, making deserialization impossible. "
                f"Please ensure that it is registered as '{config['inverse']['config']}' and identical to the original "
                "function before loading your model."
                f"{provided_function_msg}"
            )
        forward = deserialize(config["forward"], custom_objects)
        inverse = deserialize(config["inverse"], custom_objects)
        return cls(
            forward=forward,
            inverse=inverse,
        )

    def get_config(self) -> dict:
        forward_source_code = inverse_source_code = None
        try:
            forward_source_code = inspect.getsource(self._forward)
            inverse_source_code = inspect.getsource(self._inverse)
        except OSError:
            pass
        return {
            "forward": serialize(self._forward),
            "inverse": serialize(self._inverse),
            "_forward_source_code": forward_source_code,
            "_inverse_source_code": inverse_source_code,
        }

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # filter kwargs so that other transform args like batch_size, strict, ... are not passed through
        kwargs = filter_kwargs(kwargs, self._forward)
        return self._forward(data, **kwargs)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        kwargs = filter_kwargs(kwargs, self._inverse)
        return self._inverse(data, **kwargs)
