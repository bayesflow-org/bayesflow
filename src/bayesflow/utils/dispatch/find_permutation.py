import keras
from functools import singledispatch


@singledispatch
def find_permutation(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer permutation from {arg!r}.")


@find_permutation.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "random":
            from bayesflow.networks.coupling_flow.permutations import RandomPermutation

            return RandomPermutation(*args, **kwargs)
        case "swap":
            from bayesflow.networks.coupling_flow.permutations import Swap

            return Swap(*args, **kwargs)
        case "learnable" | "orthogonal":
            from bayesflow.networks.coupling_flow.permutations import OrthogonalPermutation

            return OrthogonalPermutation(*args, **kwargs)


@find_permutation.register
def _(permutation: keras.Layer, *args, **kwargs):
    return permutation


@find_permutation.register
def _(none: None, *args, **kwargs):
    return None
