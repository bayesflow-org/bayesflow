import keras

from bayesflow.utils.serialization import deserialize, serializable, serialize
from .functional import maximum_mean_discrepancy


@serializable("bayesflow.metrics")
class MaximumMeanDiscrepancy(keras.Metric):
    def __init__(
        self,
        name: str = "maximum_mean_discrepancy",
        kernel: str = "inverse_multiquadratic",
        unbiased: bool = False,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.mmd = self.add_variable(shape=(), initializer="zeros", name="mmd")
        self.kernel = kernel
        self.unbiased = unbiased

    def update_state(self, x, y):
        self.mmd.assign(
            keras.ops.cast(maximum_mean_discrepancy(x, y, kernel=self.kernel, unbiased=self.unbiased), self.dtype)
        )

    def result(self):
        return self.mmd.value

    def get_config(self):
        base_config = super().get_config()
        config = {"kernel": self.kernel, "unbiased": self.unbiased}
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
