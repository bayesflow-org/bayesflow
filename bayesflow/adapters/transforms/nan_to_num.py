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
    key : str
        The variable key to look for in the simulation data dict.
    default_value : float, optional
        Value to substitute wherever data is NaN. Default is 0.0.
    return_mask : bool, optional
        If True, a mask array will be returned under a new key. Default is False.
    mask_prefix : str, optional
        Prefix for the mask key in the output dictionary. Default is 'mask_'.
    """

    def __init__(self, key: str, default_value: float = 0.0, return_mask: bool = False, mask_prefix: str = "mask"):
        super().__init__()
        self.key = key
        self.default_value = default_value
        self.return_mask = return_mask
        self.mask_prefix = mask_prefix

    def get_config(self) -> dict:
        return serialize(
            {
                "key": self.key,
                "default_value": self.default_value,
                "return_mask": self.return_mask,
                "mask_prefix": self.mask_prefix,
            }
        )

    @property
    def mask_key(self) -> str:
        """
        Key under which the mask will be stored in the output dictionary.
        """
        return f"{self.mask_prefix}_{self.key}"

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        """
        Forward transform: fill NaNs and optionally output mask under 'mask_<key>'.
        """
        data = data.copy()

        # Check if the mask key already exists in the data
        if self.mask_key in data.keys():
            raise ValueError(
                f"Mask key '{self.mask_key}' already exists in the data. Please choose a different mask_prefix."
            )

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
            # assumes default_value is not in nan
            values[values == self.default_value] = np.nan
        else:
            mask_array = data[self.mask_key].astype(bool)
            values[~mask_array] = np.nan

        data[self.key] = values
        return data
