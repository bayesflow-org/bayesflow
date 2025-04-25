from functools import partial

import keras

from bayesflow.utils.serialization import serializable
from .functional import maximum_mean_discrepancy


@serializable
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
        self.mmd_fn = partial(maximum_mean_discrepancy, kernel=kernel, unbiased=unbiased)

    def update_state(self, x, y):
        self.mmd.assign(keras.ops.cast(self.mmd_fn(x, y), self.dtype))

    def result(self):
        return self.mmd.value
