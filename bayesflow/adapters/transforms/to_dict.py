import numpy as np
import pandas as pd

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

    def forward(self, data, **kwargs) -> dict[str, np.ndarray]:
        data = dict(data)

        for key, value in data.items():
            if isinstance(value, pd.Series):
                if value.dtype == "object":
                    value = value.astype("category")

                if value.dtype == "category":
                    value = pd.get_dummies(value)

                value = np.asarray(value).astype("float32", copy=False)

            data[key] = value

        return data

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        # non-invertible transform
        return data
