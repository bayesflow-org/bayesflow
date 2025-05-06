from collections.abc import Sequence
import numpy as np

from bayesflow.utils.serialization import serialize, serializable

from .transform import Transform


@serializable("bayesflow.adapters")
class Broadcast(Transform):
    """
    Broadcasts arrays or scalars to the shape of a given other array.

    Parameters
    ----------
    keys : sequence of str,
        Input a list of strings, where the strings are the names of data variables.
    to : str
        Name of the data variable to broadcast to.
    expand : str or int or tuple, optional
        Where should new dimensions be added to match the number of dimensions in `to`?
        Can be "left", "right", or an integer or tuple containing the indices of the new dimensions.
        The latter is needed if we want to include a dimension in the middle, which will be required
        for more advanced cases. By default we expand left.
    exclude : int or tuple, optional
        Which dimensions (of the dimensions after expansion) should retain their size,
        rather than being broadcasted to the corresponding dimension size of `to`?
        By default we exclude the last dimension (usually the data dimension) from broadcasting the size.
    squeeze : int or tuple, optional
        Axis to squeeze after broadcasting.

    Notes
    -----
    Important: Do not broadcast to variables that are used as inference variables
    (i.e., parameters to be inferred by the networks). The adapter will work during training
    but then fail during inference because the variable being broadcasted to is not available.

    Examples
    --------
    shape (1, ) array:

    >>> a = np.array((1,))

    shape (2, 3) array:

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])

    shape (2, 2, 3) array:

    >>> c = np.array([[[1, 2, 3], [4, 5, 6]], [[4, 5, 6], [1, 2, 3]]])

    >>> dat = dict(a=a, b=b, c=c)
    >>> bc = bf.adapters.transforms.Broadcast("a", to="b")
    >>> new_dat = bc.forward(dat)
    >>> new_dat["a"].shape
    (2, 1)

    >>> bc = bf.adapters.transforms.Broadcast("a", to="b", exclude=None)
    >>> new_dat = bc.forward(dat)
    >>> new_dat["a"].shape
    (2, 3)

    >>> bc = bf.adapters.transforms.Broadcast("b", to="c", expand=1)
    >>> new_dat = bc.forward(dat)
    >>> new_dat["b"].shape
    (2, 2, 3)

    It is recommended to precede this transform with a :class:`bayesflow.adapters.transforms.ToArray` transform.
    """

    def __init__(
        self,
        keys: Sequence[str],
        *,
        to: str,
        expand: str | int | tuple = "left",
        exclude: int | tuple = -1,
        squeeze: int | tuple = None,
    ):
        super().__init__()
        self.keys = keys
        self.to = to

        if isinstance(expand, int):
            expand = (expand,)

        self.expand = expand

        if isinstance(exclude, int):
            exclude = (exclude,)

        self.exclude = exclude
        self.squeeze = squeeze

    def get_config(self) -> dict:
        config = {
            "keys": self.keys,
            "to": self.to,
            "expand": self.expand,
            "exclude": self.exclude,
            "squeeze": self.squeeze,
        }
        return serialize(config)

    # noinspection PyMethodOverriding
    def forward(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        target_shape = data[self.to].shape

        data = data.copy()

        for k in self.keys:
            # ensure that .shape is defined
            data[k] = np.asarray(data[k])
            len_diff = len(target_shape) - len(data[k].shape)

            if self.expand == "left":
                data[k] = np.expand_dims(data[k], axis=tuple(np.arange(0, len_diff)))
            elif self.expand == "right":
                data[k] = np.expand_dims(data[k], axis=tuple(-np.arange(1, len_diff + 1)))
            elif isinstance(self.expand, tuple):
                if len(self.expand) is not len_diff:
                    raise ValueError("Length of `expand` must match the length difference of the involed arrays.")
                data[k] = np.expand_dims(data[k], axis=self.expand)

            new_shape = target_shape
            if self.exclude is not None:
                new_shape = np.array(new_shape, dtype=int)
                old_shape = np.array(data[k].shape, dtype=int)
                exclude = list(self.exclude)
                new_shape[exclude] = old_shape[exclude]
                new_shape = tuple(new_shape)

            data[k] = np.broadcast_to(data[k], new_shape)

            if self.squeeze is not None:
                data[k] = np.squeeze(data[k], axis=self.squeeze)

        return data

    # noinspection PyMethodOverriding
    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        # TODO: add inverse
        # we will likely never actually need the inverse broadcasting in practice
        # so adding this method is not high priority
        return data
