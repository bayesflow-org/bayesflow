r"""
Elementwise bijections applied by the coupling layers of a
:py:class:`~bayesflow.networks.CouplingFlow`.
"""

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


from bayesflow.utils._docs import _add_imports_to_all  # noqa: E402

# TODO: move the helper to bayesflow.utils.find_transform
_add_imports_to_all(exclude=["find_transform"])
