import numpy as np

from bayesflow.utils.serialization import serializable, serialize
from .elementwise_transform import ElementwiseTransform


@serializable
class ReplaceNaN(ElementwiseTransform):
    """
    Replace NaNs with a default value, and optionally encode a missing‐data mask.

    This is based on "Missing data in amortized simulation-based neural posterior estimation" by Wang et al. (2024).

    Parameters
    ----------
    default_value : float
        Value to substitute wherever data is NaN.
    encode_mask : bool, default=False
        If True, the forward pass will expand the array by one new axis and
        concatenate a binary mask (0 for originally-NaN entries, 1 otherwise).
    axis : int or None
        Axis along which to add the new dimension for mask encoding.
        If None, defaults to `data.ndim` (i.e., a new trailing axis).

    Examples
    --------
    >>> a = np.array([1.0, np.nan, 3.0])
    >>> r_nan = bf.adapters.transforms.ReplaceNaN(default_value=0.0)
    >>> r_nan.forward(a)
    array([1., 0., 3.])

    >>> # With mask encoding along a new last axis:
    >>> r_nan = bf.adapters.transforms.ReplaceNaN(default_value=-1.0, encode_mask=True, axis=-1)
    >>> enc = r_nan.forward(a)
    >>> enc.shape
    (3, 2)

    It’s recommended to precede this with a ToArray transform if your data
    might not already be a NumPy array.
    """

    def __init__(
        self,
        *,
        default_value: float = 0.0,
        encode_mask: bool = False,
        axis: int | None = None,
    ):
        super().__init__()
        self.default_value = default_value
        self.encode_mask = encode_mask
        self.axis = axis

    def get_config(self) -> dict:
        return serialize(
            {
                "default_value": self.default_value,
                "encode_mask": self.encode_mask,
                "axis": self.axis,
            }
        )

    def forward(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # Create mask of where data is NaN
        mask = np.isnan(data)
        # Fill NaNs with the default value
        filled = np.where(mask, self.default_value, data)

        if not self.encode_mask:
            return filled

        # Decide where to insert the new axis
        ax = self.axis if self.axis is not None else data.ndim
        # Expand dims for both filled data and mask
        filled_exp = np.expand_dims(filled, axis=ax)
        mask_exp = 1 - np.expand_dims(mask.astype(np.int8), axis=ax)
        # Concatenate along that axis: [..., value, mask]
        return np.concatenate([filled_exp, mask_exp], axis=ax)

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        if not self.encode_mask:
            # No mask was encoded, so nothing to undo
            return data

        ax = self.axis if self.axis is not None else data.ndim - 1
        # Extract the two “channels”
        values = np.take(data, indices=0, axis=ax)
        mask = np.take(data, indices=1, axis=ax).astype(bool)
        # Restore NaNs where mask == 1
        values[mask] = np.nan
        return values
