from functools import singledispatch

from bayesflow.distributions import Distribution


@singledispatch
def find_distribution(arg, **kwargs):
    raise TypeError(f"Cannot infer distribution from {arg!r}.")


@find_distribution.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "normal":
            from bayesflow.distributions import DiagonalNormal

            distribution = DiagonalNormal(*args, **kwargs)
        case "none":
            distribution = None
        case other:
            raise ValueError(f"Unsupported distribution name '{other}'.")

    return distribution


@find_distribution.register
def _(none: None, *args, **kwargs):
    return None


@find_distribution.register
def _(distribution: Distribution, *args, **kwargs):
    return distribution
