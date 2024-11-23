from collections.abc import Sequence
import numpy as np

from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transform import Transform


@serializable(package="bayesflow.adapters")
class Broadcast(Transform):
    """
    Broadcasts arrays or scalars to the shape of a given other array.

    Parameters:

    expand: Where should new dimensions be added to match the number of dimensions in `to`?
    Can be "left", "right", or an integer or tuple containing the indices of the new dimensions.
    The latter is needed if we want to include a dimension in the middle, which will be required
    for more advanced cases. By default we expand left.

    exclude: Which dimensions (of the dimensions after expansion) should retain their size,
    rather than being broadcasted to the corresponding dimension size of `to`?
    By default we exclude the last dimension (usually the data dimension) from broadcasting the size.

    Examples:
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

    def __init__(self, keys: Sequence[str], *, to: str, expand: str | int | tuple = "left", exclude: int | tuple = -1):
        super().__init__()
        self.keys = keys
        self.to = to

        if isinstance(expand, int):
            expand = (expand,)

        self.expand = expand

        if isinstance(exclude, int):
            exclude = (exclude,)

        self.exclude = exclude

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Broadcast":
        return cls(
            keys=deserialize(config["keys"], custom_objects),
            to=deserialize(config["to"], custom_objects),
            expand=deserialize(config["expand"], custom_objects),
            exclude=deserialize(config["exclude"], custom_objects),
        )

    def get_config(self) -> dict:
        return {
            "keys": serialize(self.keys),
            "to": serialize(self.to),
            "expand": serialize(self.expand),
            "exclude": serialize(self.exclude),
        }

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

        return data

    # noinspection PyMethodOverriding
    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        # TODO: add inverse
        # we will likely never actually need the inverse broadcasting in practice
        # so adding this method is not high priority
        return data
