from collections.abc import Sequence

from bayesflow.utils.serialization import serializable, serialize

from .transform import Transform


@serializable("bayesflow.adapters")
class Drop(Transform):
    """
    Transform to drop variables from further calculation.

    Parameters
    ----------
    keys : sequence of str
        Names of data variables that should be dropped

    Examples
    --------

    >>> import bayesflow as bf
    >>> a = [1, 2, 3, 4]
    >>> b = [[1, 2], [3, 4]]
    >>> c = [[5, 6, 7, 8]]
    >>> dat = dict(a=a, b=b, c=c)
    >>> dat
        {'a': [1, 2, 3, 4], 'b': [[1, 2], [3, 4]], 'c': [[5, 6, 7, 8]]}
    >>> drop = bf.adapters.transforms.Drop(("b", "c"))
    >>> drop.forward(dat)
        {'a': [1, 2, 3, 4]}
    """

    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def get_config(self) -> dict:
        return serialize({"keys": self.keys})

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        # no strict version because there is no requirement for the keys to be present
        return {key: value for key, value in data.items() if key not in self.keys}

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        # non-invertible transform
        return data

    def extra_repr(self) -> str:
        return "[" + ", ".join(map(repr, self.keys)) + "]"

    def log_det_jac(self, data: dict[str, any], log_det_jac: dict[str, any], inverse: bool = False, **kwargs):
        return self.inverse(data=log_det_jac) if inverse else self.forward(data=log_det_jac)
