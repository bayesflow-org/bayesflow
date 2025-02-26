from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from .elementwise_transform import ElementwiseTransform
from .transform import Transform


class Predicate(Protocol):
    def __call__(self, key: str, value: np.ndarray, inverse: bool) -> bool:
        raise NotImplementedError


class FilterTransform(Transform):
    """
    Implements a transform that applies a different transform on a subset of the data. Used by other transforms and
    base adapter class.
    """

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
        self.initialize_config()

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

    def __repr__(self):
        if e := self.extra_repr():
            return f"{self.transform_constructor.__name__}({e})"

        return self.transform_constructor.__name__

    def extra_repr(self) -> str:
        result = ""

        if self.predicate is not None:
            result += f"predicate={self.predicate.__name__}"

        if self.include is not None:
            if result:
                result += ", "

            result += f"include={self.include!r}"

        if self.exclude is not None:
            if result:
                result += ", "

            result += f"exclude={self.exclude!r}"

        return result

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
        if self.predicate:
            if self.exclude and key in self.exclude:
                return False
            if self.include and key in self.include:
                return True
            return self.predicate(key, value, inverse=inverse)
        else:
            if self.include and key in self.include:
                return True
            if self.exclude and key in self.exclude:
                return False
            return not self.include and not self.exclude

    def _apply_transform(self, key: str, value: np.ndarray, inverse: bool = False, **kwargs) -> np.ndarray:
        if key not in self.transform_map:
            self.transform_map[key] = self.transform_constructor(**self.kwargs)

        transform = self.transform_map[key]

        return transform(value, inverse=inverse, **kwargs)
