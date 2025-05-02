from collections.abc import Sequence

from bayesflow.utils.serialization import serializable, serialize

from .transform import Transform


@serializable("bayesflow.adapters")
class Keep(Transform):
    """
    Name the data parameters that should be kept for futher calculation.

    Parameters
    ----------
    keys : sequence of str
        The names of kept data variables as strings.

    Examples
    --------

    Two moons simulator generates data for priors alpha, r and theta as well as observation data x.
    We are interested only in theta and x, to keep only theta and x we should use the following;

    >>> adapter = (
        bf.adapters.Adapter()
            # drop data from unneeded priors alpha, and r
            # only keep theta and x
            .keep(("theta", "x"))
        )

    The following example shows the usage in a more general case:

    >>> a = [1, 2, 3, 4]
    >>> b = [[1, 2], [3, 4]]
    >>> c = [[5, 6, 7, 8]]
    >>> dat = dict(a=a, b=b, c=c)

    Here we want to only keep elements b and c

    >>> keeper = bf.adapters.transforms.Keep(("b", "c"))
    >>> keeper.forward(dat)
    {'b': [[1, 2], [3, 4]], 'c': [[5, 6, 7, 8]]}
    """

    def __init__(self, keys: Sequence[str]):
        self.keys = keys

    def get_config(self) -> dict:
        return serialize({"keys": self.keys})

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        return {key: value for key, value in data.items() if key in self.keys}

    def inverse(self, data: dict[str, any], **kwargs) -> dict[str, any]:
        # non-invertible transform
        return data

    def extra_repr(self) -> str:
        return "[" + ", ".join(map(repr, self.keys)) + "]"

    def log_det_jac(self, data: dict[str, any], log_det_jac: dict[str, any], inverse: bool = False, **kwargs):
        return self.inverse(data=log_det_jac) if inverse else self.forward(data=log_det_jac)
