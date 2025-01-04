from collections.abc import Callable, Sequence

import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transforms import (
    AsSet,
    Broadcast,
    Concatenate,
    Constrain,
    ConvertDType,
    Drop,
    ExpandDims,
    FilterTransform,
    Keep,
    LambdaTransform,
    MapTransform,
    OneHot,
    Rename,
    Standardize,
    ToArray,
    Transform,
)

from .transforms.filter_transform import Predicate


@serializable(package="bayesflow.adapters")
class Adapter:
    def __init__(self, transforms: Sequence[Transform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = transforms

    @staticmethod
    def create_default(inference_variables: Sequence[str]) -> "Adapter":
        return (
            Adapter()
            .to_array()
            .convert_dtype("float64", "float32")
            .concatenate(inference_variables, into="inference_variables")
        )

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Adapter":
        return cls(transforms=deserialize(config["transforms"], custom_objects))

    def get_config(self) -> dict:
        return {"transforms": serialize(self.transforms)}

    def forward(self, data: dict[str, any], **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        for transform in self.transforms:
            data = transform(data, **kwargs)

        return data

    def inverse(self, data: dict[str, np.ndarray], **kwargs) -> dict[str, any]:
        data = data.copy()

        for transform in reversed(self.transforms):
            data = transform(data, inverse=True, **kwargs)

        return data

    def __call__(self, data: dict[str, any], *, inverse: bool = False, **kwargs) -> dict[str, np.ndarray]:
        if inverse:
            return self.inverse(data, **kwargs)

        return self.forward(data, **kwargs)

    def __repr__(self):
        str_transf = ""
        if isinstance(self.transforms, list):
            for i in range(0, len(self.transforms)):
                str_transf = str_transf + str(i) + ": " + repr(self.transforms[i])
                if i != len(self.transforms) - 1:
                    str_transf = str_transf + " -> "
            return f"Adapter([{str_transf}])"
        else:
            return f"Adapter([ 0: {repr(self.transforms)}])"

    def __getitem__(self, index):
        return Adapter(transforms=self.transforms[index])

    def __setitem__(self, index, new_value):
        if not isinstance(new_value, Adapter):
            raise TypeError("new_value must be an Adapter instance")

        # new_transform = new_value.transforms

        # To be tested
        if len(new_value.transforms) == 0:
            raise ValueError(
                "new_value is an Adapter instance without any specified transforms, new_value Adapter must contain at least one transform."
            )

        if isinstance(index, slice):
            self.transforms[index] = new_value.transforms[:]

        elif isinstance(index, int):
            self.transforms[index : index + 1] = new_value.transforms[:]

        else:
            raise TypeError("Invalid index type. Must be int or slice.")

    def add_transform(self, transform: Transform):
        self.transforms.append(transform)
        return self

    def apply(
        self,
        *,
        forward: Callable[[np.ndarray, ...], np.ndarray],
        inverse: Callable[[np.ndarray, ...], np.ndarray],
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        transform = FilterTransform(
            transform_constructor=LambdaTransform,
            predicate=predicate,
            include=include,
            exclude=exclude,
            forward=forward,
            inverse=inverse,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    # Begin of transforms derived from transform classes
    def as_set(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsSet() for key in keys})
        self.transforms.append(transform)
        return self

    def broadcast(
        self, keys: str | Sequence[str], *, to: str, expand: str | int | tuple = "left", exclude: int | tuple = -1
    ):
        if isinstance(keys, str):
            keys = [keys]

        transform = Broadcast(keys, to=to, expand=expand, exclude=exclude)
        self.transforms.append(transform)
        return self

    def clear(self):
        self.transforms = []
        return self

    def concatenate(self, keys: Sequence[str], *, into: str, axis: int = -1):
        if isinstance(keys, str):
            # this is a common mistake, and also passes the type checker since str is a sequence of characters
            raise ValueError("Keys must be a sequence of strings. To rename a single key, use the `rename` method.")

        transform = Concatenate(keys, into=into, axis=axis)
        self.transforms.append(transform)
        return self

    def convert_dtype(
        self,
        from_dtype: str,
        to_dtype: str,
        *,
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
    ):
        transform = FilterTransform(
            transform_constructor=ConvertDType,
            predicate=predicate,
            include=include,
            exclude=exclude,
            from_dtype=from_dtype,
            to_dtype=to_dtype,
        )
        self.transforms.append(transform)
        return self

    def constrain(
        self,
        keys: str | Sequence[str],
        *,
        lower: int | float | np.ndarray = None,
        upper: int | float | np.ndarray = None,
        method: str = "default",
    ):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform(
            transform_map={key: Constrain(lower=lower, upper=upper, method=method) for key in keys}
        )
        self.transforms.append(transform)
        return self

    def drop(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = Drop(keys)
        self.transforms.append(transform)
        return self

    def expand_dims(self, keys: str | Sequence[str], *, axis: int | tuple):
        if isinstance(keys, str):
            keys = [keys]

        transform = ExpandDims(keys, axis=axis)
        self.transforms.append(transform)
        return self

    def keep(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = Keep(keys)
        self.transforms.append(transform)
        return self

    def one_hot(self, keys: str | Sequence[str], num_classes: int):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: OneHot(num_classes=num_classes) for key in keys})
        self.transforms.append(transform)
        return self

    def rename(self, from_key: str, to_key: str):
        self.transforms.append(Rename(from_key, to_key))
        return self

    def standardize(
        self,
        *,
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        transform = FilterTransform(
            transform_constructor=Standardize,
            predicate=predicate,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self.transforms.append(transform)
        return self

    def to_array(
        self,
        *,
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        transform = FilterTransform(
            transform_constructor=ToArray,
            predicate=predicate,
            include=include,
            exclude=exclude,
            **kwargs,
        )
        self.transforms.append(transform)
        return self
