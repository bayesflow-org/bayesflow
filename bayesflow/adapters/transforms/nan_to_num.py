import numpy as np

from bayesflow.utils.serialization import serializable, serialize
from .transform import Transform


@serializable("bayesflow.adapters")
class NanToNum(Transform):
    """
    Replace NaNs with a default value, and optionally encode a missing-data mask as a separate output key.

    This is based on "Missing data in amortized simulation-based neural posterior estimation" by Wang et al. (2024).

    Parameters
    ----------
    default_value : float
        Value to substitute wherever data is NaN.
    return_mask : bool, default=False
        If True, a mask array will be returned under a new key.
    """

    def __init__(self, key: str, default_value: float = 0.0, return_mask: bool = False):
        super().__init__()
        self.key = key
        self.default_value = default_value
        self.return_mask = return_mask

    def get_config(self) -> dict:
        return serialize(
            {
                "key": self.key,
                "default_value": self.default_value,
                "return_mask": self.return_mask,
            }
        )

    @property
    def mask_key(self) -> str:
        """
        Key under which the mask will be stored in the output dictionary.
        """
        return f"mask_{self.key}" if self.key else "mask"

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        """
        Forward transform: fill NaNs and optionally output mask under 'mask_<key>'.
        """
        data = data.copy()

        # Identify NaNs and fill with default value
        mask = np.isnan(data[self.key])
        data[self.key] = np.nan_to_num(data[self.key], copy=False, nan=self.default_value)

        if not self.return_mask:
            return data

        # Prepare mask array (1 for valid, 0 for NaN)
        mask_array = (~mask).astype(np.int8)

        # Return both the filled data and the mask under separate keys
        data[self.mask_key] = mask_array
        return data

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        """
        Inverse transform: restore NaNs using the mask under 'mask_<key>'.
        """
        data = data.copy()

        # Retrieve mask and values to reconstruct NaNs
        values = data[self.key]

        if not self.return_mask:
            values[values == self.default_value] = np.nan  # we assume default_value is not in data
        else:
            mask_array = data[self.mask_key].astype(bool)
            # Put NaNs where mask is 0
            values[~mask_array] = np.nan

        data[self.key] = values
        return data
