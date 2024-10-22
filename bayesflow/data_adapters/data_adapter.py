from collections.abc import Sequence

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


@serializable(package="bayesflow.data_adapters")
class DataAdapter:
    def __init__(self, transforms: Sequence[Transform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = transforms

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "DataAdapter":
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

    def add_transform(self, transform: Transform):
        self.transforms.append(transform)
        return self

    def apply(
        self,
        *,
        forward: callable,
        inverse: callable,
        predicate: callable = None,
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

    def as_set(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsSet() for key in keys})
        self.transforms.append(transform)
        return self

    def broadcast(self, keys: str | Sequence[str], *, expand_scalars: bool = True):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: Broadcast(expand_scalars=expand_scalars) for key in keys})
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
        predicate: callable = None,
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
        predicate: callable = None,
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
        predicate: callable = None,
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
