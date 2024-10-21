from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from .elementwise_transform import ElementwiseTransform
from .transform import Transform


class Predicate(Protocol):
    def __call__(self, key: str, value: np.ndarray, inverse: bool) -> bool:
        raise NotImplementedError


@serializable(package="bayesflow.data_adapters")
class FilterTransform(Transform):
    def __init__(
        self,
        *,
        transform_constructor: Callable[..., ElementwiseTransform],
        predicate: Predicate = None,
        include: str | Sequence[str] = None,
        exclude: str | Sequence[str] = None,
        **kwargs,
    ):
        super().__init__()

        if isinstance(include, str):
            include = [include]

        if isinstance(exclude, str):
            exclude = [exclude]

        self.transform_constructor = transform_constructor

        self.predicate = predicate
        self.include = include
        self.exclude = exclude

        self.kwargs = kwargs

        self.transform_map = {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Transform":
        def transform_constructor(*args, **kwargs):
            raise RuntimeError(
                "Instantiating new elementwise transforms on a deserialized FilterTransform is not yet supported (and"
                "may never be). As a work-around, you can manually register the elementwise transform constructor after"
                "deserialization:\n"
                "obj = deserialize(config)\n"
                "obj.transform_constructor = MyElementwiseTransform"
            )

        instance = cls(
            transform_constructor=transform_constructor,
            predicate=deserialize(config["predicate"], custom_objects),
            include=deserialize(config["include"], custom_objects),
            exclude=deserialize(config["exclude"], custom_objects),
            **config["kwargs"],
        )

        instance.transform_map = deserialize(config["transform_map"])

        return instance

    def get_config(self) -> dict:
        return {
            "predicate": serialize(self.predicate),
            "include": serialize(self.include),
            "exclude": serialize(self.exclude),
            "kwargs": serialize(self.kwargs),
            "transform_map": serialize(self.transform_map),
        }

    def forward(self, data: dict[str, np.ndarray], *, strict: bool = True, **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        if strict and self.include is not None:
            missing_keys = set(self.include) - set(data.keys())
            if missing_keys:
                raise KeyError(f"Missing keys from include list: {missing_keys!r}")

        for key, value in data.items():
            if self._should_transform(key, value, inverse=False):
                data[key] = self._apply_transform(key, value, inverse=False, **kwargs)

        return data

    def inverse(self, data: dict[str, np.ndarray], *, strict: bool = False, **kwargs) -> dict[str, np.ndarray]:
        data = data.copy()

        if strict and self.include is not None:
            missing_keys = set(self.include) - set(data.keys())
            if missing_keys:
                raise KeyError(f"Missing keys from include list: {missing_keys!r}")

        for key, value in data.items():
            if self._should_transform(key, value, inverse=True):
                data[key] = self._apply_transform(key, value, inverse=True)

        return data

    def _should_transform(self, key: str, value: np.ndarray, inverse: bool = False) -> bool:
        match self.predicate, self.include, self.exclude:
            case None, None, None:
                return True

            case None, None, exclude:
                return key not in exclude

            case None, include, None:
                return key in include

            case None, include, exclude:
                return key in include and key not in exclude

            case predicate, None, None:
                return predicate(key, value, inverse=inverse)

            case predicate, None, exclude:
                if key in exclude:
                    return False
                return predicate(key, value, inverse=inverse)

            case predicate, include, None:
                if key in include:
                    return True
                return predicate(key, value, inverse=inverse)

            case predicate, include, exclude:
                if key in exclude:
                    return False
                if key in include:
                    return True
                return predicate(key, value, inverse=inverse)

    def _apply_transform(self, key: str, value: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        if key not in self.transform_map:
            self.transform_map[key] = self.transform_constructor(**self.kwargs)

        transform = self.transform_map[key]

        return transform(value, inverse=inverse, **kwargs)
