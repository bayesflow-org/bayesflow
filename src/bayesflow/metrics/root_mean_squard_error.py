import keras

from bayesflow.utils.serialization import deserialize, serializable
from .functional import root_mean_squared_error


@serializable("bayesflow.metrics")
class RootMeanSquaredError(keras.metrics.MeanMetricWrapper):
    def __init__(self, name="root_mean_squared_error", dtype=None, **kwargs):
        super().__init__(root_mean_squared_error, name=name, dtype=dtype, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        # fn is fixed and passed directly in the constructor
        base_config.pop("fn")
        return base_config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
