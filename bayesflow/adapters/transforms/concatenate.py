from collections.abc import Sequence

import numpy as np

from bayesflow.utils.serialization import serialize, serializable

from .transform import Transform


@serializable("bayesflow.adapters")
class Concatenate(Transform):
    """Concatenate multiple arrays into a new key. Used to specify how data variables should be treated by the network.

    Parameters
    ----------
    keys : sequence of str,
        Input a list of strings, where the strings are the names of data variables.
    into : str
        A string telling the network how to use the variables named in keys.
    axis : int, optional
        Along which axis to concatenate the keys. The last axis is used by default.

    Examples
    --------
    Suppose you have a simulator that generates variables "beta" and "sigma" from priors and then observation
    variables "x" and "y". We can then use concatenate in the following way

    >>> adapter = (
        bf.Adapter()
            .concatenate(["beta", "sigma"], into="inference_variables")
            .concatenate(["x", "y"], into="summary_variables")
    )
    """

    def __init__(self, keys: Sequence[str], *, into: str, axis: int = -1, indices: list | None = None):
        self.keys = keys
        self.into = into
        self.axis = axis

        self.indices = indices

    def get_config(self) -> dict:
        config = {
            "keys": self.keys,
            "into": self.into,
            "axis": self.axis,
            "indices": self.indices,
        }
        return serialize(config)

    def forward(self, data: dict[str, any], *, strict: bool = True, **kwargs) -> dict[str, any]:
        if not strict and self.indices is None:
            raise ValueError("Cannot call `forward` with `strict=False` before calling `forward` with `strict=True`.")

        # copy to avoid side effects
        data = data.copy()

        required_keys = set(self.keys)
        available_keys = set(data.keys())
        common_keys = available_keys & required_keys
        missing_keys = required_keys - available_keys

        if strict and missing_keys:
            # invalid call
            raise KeyError(f"Missing keys: {missing_keys!r}")
        elif missing_keys:
            # we cannot produce a result, but should still remove the keys
            for key in common_keys:
                data.pop(key)

            return data

        if self.indices is None:
            # remember the indices of the parts in the concatenated array
            self.indices = np.cumsum([data[key].shape[self.axis] for key in self.keys]).tolist()

        # remove each part
        parts = [data.pop(key) for key in self.keys]

        # concatenate them all
        result = np.concatenate(parts, axis=self.axis)

        # store the result
        data[self.into] = result

        return data

    def inverse(self, data: dict[str, any], *, strict: bool = False, **kwargs) -> dict[str, any]:
        if self.indices is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")

        # copy to avoid side effects
        data = data.copy()

        if strict and self.into not in data:
            # invalid call
            raise KeyError(f"Missing key: {self.into!r}")
        elif self.into not in data:
            # nothing to do
            return data

        # split the concatenated array and remove the concatenated key
        keys = self.keys
        values = np.split(data.pop(self.into), self.indices, axis=self.axis)

        # restore the parts
        data |= dict(zip(keys, values))

        return data

    def extra_repr(self) -> str:
        result = "[" + ", ".join(map(repr, self.keys)) + "] -> " + repr(self.into)

        if self.axis != -1:
            result += f", axis={self.axis}"

        return result

    def log_det_jac(
        self,
        data: dict[str, np.ndarray],
        log_det_jac: dict[str, np.ndarray],
        *,
        strict: bool = False,
        inverse: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        # copy to avoid side effects
        log_det_jac = log_det_jac.copy()

        if inverse:
            if log_det_jac.get(self.into) is not None:
                raise ValueError(
                    "Cannot obtain an inverse Jacobian of concatenation. "
                    "Transform your variables before you concatenate."
                )

            return log_det_jac

        required_keys = set(self.keys)
        available_keys = set(log_det_jac.keys())
        common_keys = available_keys & required_keys

        if len(common_keys) == 0:
            return log_det_jac

        parts = [log_det_jac.pop(key) for key in common_keys]

        log_det_jac[self.into] = sum(parts)

        return log_det_jac
