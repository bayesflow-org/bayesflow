from collections.abc import Callable, MutableSequence, Sequence

import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .transforms import (
    AsSet,
    AsTimeSeries,
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
class Adapter(MutableSequence[Transform]):
    def __init__(self, transforms: Sequence[Transform] | None = None):
        if transforms is None:
            transforms = []

        self.transforms = list(transforms)

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
        result = ""
        for i, transform in enumerate(self):
            result += f"{i}: {transform!r}"
            if i != len(self) - 1:
                result += " -> "

        return f"Adapter([{result}])"

    # list methods

    def append(self, value: Transform) -> "Adapter":
        self.transforms.append(value)
        return self

    def __delitem__(self, key: int | slice):
        del self.transforms[key]

    def extend(self, values: Sequence[Transform]) -> "Adapter":
        if isinstance(values, Adapter):
            values = values.transforms

        self.transforms.extend(values)

        return self

    def __getitem__(self, item: int | slice) -> "Adapter":
        if isinstance(item, int):
            return self.transforms[item]

        return Adapter(self.transforms[item])

    def insert(self, index: int, value: Transform | Sequence[Transform]) -> "Adapter":
        if isinstance(value, Adapter):
            value = value.transforms

        if isinstance(value, Sequence):
            # convenience: Adapters are always flat
            self.transforms = self.transforms[:index] + list(value) + self.transforms[index:]
        else:
            self.transforms.insert(index, value)

        return self

    def __setitem__(self, key: int | slice, value: Transform | Sequence[Transform]) -> "Adapter":
        if isinstance(value, Adapter):
            value = value.transforms

        if isinstance(key, int) and isinstance(value, Sequence):
            if key < 0:
                key += len(self.transforms)

            key = slice(key, key + 1)

        self.transforms[key] = value

        return self

    def __len__(self):
        return len(self.transforms)

    # adapter methods

    add_transform = append

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


    def as_set(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsSet() for key in keys})
        self.transforms.append(transform)
        return self

    def as_time_series(self, keys: str | Sequence[str]):
        if isinstance(keys, str):
            keys = [keys]

        transform = MapTransform({key: AsTimeSeries() for key in keys})
        self.transforms.append(transform)
        return self

    def broadcast(
        self,
        keys: str | Sequence[str],
        *,
        to: str,
        expand: str | int | tuple = "left",
        exclude: int | tuple = -1,
        squeeze: int | tuple = None,
    ):
        if isinstance(keys, str):
            keys = [keys]

        transform = Broadcast(keys, to=to, expand=expand, exclude=exclude, squeeze=squeeze)
        self.transforms.append(transform)
        return self

    def clear(self):
        self.transforms = []
        return self

    def concatenate(self, keys: str | Sequence[str], *, into: str, axis: int = -1):
        if isinstance(keys, str):
            transform = Rename(keys, to_key=into)
        else:
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
