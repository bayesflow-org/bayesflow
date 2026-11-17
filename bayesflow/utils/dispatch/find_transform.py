import keras
from functools import singledispatch


@singledispatch
def find_transform(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer transform from {arg!r}.")


@find_transform.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "affine":
            from bayesflow.networks.inference.coupling.transforms import AffineTransform

            return AffineTransform()
        case "spline":
            from bayesflow.networks.inference.coupling.transforms import SplineTransform

            return SplineTransform(*args, **kwargs)
        case str() as unknown_transform:
            raise ValueError(f"Unknown transform: '{unknown_transform}'")


@find_transform.register
def _(transform: type, *args, **kwargs):
    return transform()


@find_transform.register
def _(transform: keras.Layer, *args, **kwargs):
    return transform
