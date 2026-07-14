import pandas as pd
import keras.ops as ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable

from .transform import Transform


@serializable("bayesflow.adapters")
class ToDict(Transform):
    """Convert non-dict batches (e.g., pandas.DataFrame) to dict batches"""

    @classmethod
    def from_config(cls, config: dict, custom_objects=None):
        return cls()

    def get_config(self) -> dict:
        return {}

    def forward(self, data, **kwargs) -> dict[str, Tensor]:
        data = dict(data)

        for key, value in data.items():
            if isinstance(value, pd.Series):
                if value.dtype == "object":
                    value = value.astype("category")

                if value.dtype == "category":
                    value = pd.get_dummies(value)

                value = ops.convert_to_tensor(value)

            data[key] = value

        return data

    def inverse(self, data: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        # non-invertible transform
        return data
