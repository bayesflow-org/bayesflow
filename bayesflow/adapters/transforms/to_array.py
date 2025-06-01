from numbers import Number

import numpy as np

from bayesflow.utils.tree import map_dict, get_value_at_path, map_dict_with_path
from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class ToArray(ElementwiseTransform):
    """
    Checks provided data for any non-arrays and converts them to numpy arrays.

    This ensures all data is in a format suitable for training.

    Examples
    --------
    >>> ta = bf.adapters.transforms.ToArray()
    >>> a = [1, 2, 3, 4]
    >>> ta.forward(a)
        array([1, 2, 3, 4])
    >>> b = [[1, 2], [3, 4]]
    >>> ta.forward(b)
        array([[1, 2],
            [3, 4]])
    """

    def __init__(self, original_type: type = None):
        super().__init__()
        self.original_type = original_type

    def get_config(self) -> dict:
        return serialize({"original_type": self.original_type})

    def forward(self, data: any, **kwargs) -> np.ndarray:
        if self.original_type is None:
            if isinstance(data, dict):
                self.original_type = map_dict(type, data)
            else:
                self.original_type = type(data)

        if isinstance(self.original_type, dict):
            # use self.original_type in check to preserve serializablitiy
            return map_dict(np.asarray, data)
        return np.asarray(data)

    def inverse(self, data: np.ndarray | dict, **kwargs) -> any:
        if self.original_type is None:
            raise RuntimeError("Cannot call `inverse` before calling `forward` at least once.")
        if isinstance(self.original_type, dict):
            # use self.original_type in check to preserve serializablitiy

            def restore_original_type(path, value):
                try:
                    original_type = get_value_at_path(self.original_type, path)
                    return original_type(value)
                except KeyError:
                    pass
                except TypeError:
                    pass
                except ValueError:
                    # separate statements, as optree does not allow (KeyError | TypeError | ValueError)
                    pass
                return value

            return map_dict_with_path(restore_original_type, data)

        if issubclass(self.original_type, Number):
            try:
                return self.original_type(data.item())
            except ValueError:
                pass

        # cannot invert
        return data
