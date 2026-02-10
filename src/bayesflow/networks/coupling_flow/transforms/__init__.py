from .affine_transform import AffineTransform
from .spline_transform import SplineTransform
from .transform import Transform


def find_transform(transform: str | Transform | type(Transform), **kwargs) -> Transform:
    if isinstance(transform, Transform):
        return transform
    if isinstance(transform, type):
        return transform()

    match transform.lower():
        case "affine":
            return AffineTransform()
        case "spline":
            return SplineTransform(**kwargs)
        case str() as unknown_transform:
            raise ValueError(f"Unknown transform: '{unknown_transform}'")
        case other:
            raise TypeError(f"Unknown transform type: {other}")
