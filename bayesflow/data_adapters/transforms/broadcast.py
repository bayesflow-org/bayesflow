import numpy as np

from .elementwise_transform import ElementwiseTransform


class Broadcast(ElementwiseTransform):
    """
    Broadcasts array to a given batch size.
    Examples:
        >>> bc = Broadcast()
        >>> bc(np.array(5), batch_size=3)
        array([[5], [5], [5]])
        >>> bc(np.array(5), batch_size=3).shape
        (3, 1)
        >>> bc(np.array([1, 2, 3]), batch_size=3)
        array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        >>> bc(np.array([1, 2, 3]), batch_size=3).shape
        (3, 3)

        You can opt out of expanding scalars:
        >>> bc = Broadcast(expand_scalars=False)
        >>> bc(np.array(5), batch_size=3)
        np.array([5, 5, 5])
        >>> bc(np.array(5), batch_size=3).shape
        (3,)

    It is recommended to precede this transform with a :class:`bayesflow.data_adapters.transforms.ToArray` transform.
    """

    def __init__(self, *, expand_scalars: bool = True):
        super().__init__()

        self.expand_scalars = expand_scalars

    # noinspection PyMethodOverriding
    def forward(self, data: np.ndarray, *, batch_size: int, **kwargs):
        data = np.repeat(data[None], batch_size, axis=0)

        if self.expand_scalars and data.ndim == 1:
            data = data[:, None]

        return data

    # noinspection PyMethodOverriding
    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        data = data[0]

        if self.expand_scalars:
            data = np.squeeze(data, axis=0)

        return data
