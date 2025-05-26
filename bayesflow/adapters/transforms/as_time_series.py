import numpy as np

from bayesflow.utils.serialization import serializable

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class AsTimeSeries(ElementwiseTransform):
    """The `.as_time_series` transform can be used to indicate that variables shall be treated as time series.

    Currently, all this transformation does is to ensure that the variable
    arrays are at least 3D. The 2rd dimension is treated as the
    time series dimension and the 3rd dimension as the data dimension.
    In the future, the transform will have more advanced behavior
    to better ensure the correct treatment of time series data.

    Examples
    --------

    >>> adapter = bf.Adapter().as_time_series(["x", "y"])
    """

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return np.atleast_3d(data)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if data.shape[2] == 1:
            return np.squeeze(data, axis=2)

        return data

    def get_config(self) -> dict:
        return {}
