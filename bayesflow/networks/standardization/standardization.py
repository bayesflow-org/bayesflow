from collections.abc import Sequence

import keras

from bayesflow.utils.serialization import serializable, serialize, deserialize
from bayesflow.utils import layer_kwargs
from bayesflow.types import Tensor

from .standardize import Standardize


@serializable("bayesflow.networks")
class Standardization(keras.Layer):
    """
    Initializes a layer that will keep track of the running mean and
    running standard deviation for all possible variables in a BayesFlow
    approximator.
    """

    def __init__(self, standardize: str | Sequence | None, **kwargs):
        super().__init__(**layer_kwargs(kwargs))

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            # we have to lazily initialize these
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardize(trainable=False) for var in self.standardize}

    def maybe_standardize(
        self,
        x: Tensor | None,
        key: str,
        stage: str = "inference",
        forward: bool = True,
        log_det_jac: bool = False,
        transformation_type: str = "location_scale",
    ):
        if x is None:
            if log_det_jac:
                return None, 0.0
            return

        if key not in self.standardize:
            if log_det_jac:
                return x, 0.0
            return x

        return self.standardize_layers[key](
            x, forward=forward, stage=stage, log_det_jac=log_det_jac, transformation_type=transformation_type
        )

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        if self.standardize == "all":
            # Only include variables present in data_shapes
            self.standardize = [
                var
                for var in ["inference_variables", "summary_variables", "inference_conditions"]
                if var in data_shapes
            ]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        for var, layer in self.standardize_layers.items():
            layer.build(data_shapes[var])

        self.built = True

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)
        config = {"standardize": self.standardize}
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
