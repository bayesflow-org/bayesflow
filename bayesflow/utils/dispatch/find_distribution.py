from functools import singledispatch
import keras


@singledispatch
def find_distribution(arg, **kwargs):
    raise TypeError(f"Cannot infer distribution from {arg!r}.")


@find_distribution.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "normal":
            from bayesflow.distributions import DiagonalNormal

            distribution = DiagonalNormal(*args, **kwargs)

        case "student" | "student-t" | "student_t":
            from bayesflow.distributions import DiagonalStudentT

            distribution = DiagonalStudentT(*args, **kwargs)

        case "mixture":
            raise ValueError(
                "Mixture distributions need to be explicitly defined as bf.distributions.Mixture(...) "
                "and passed to the constructor."
            )
        case "none":
            distribution = None

        case other:
            raise ValueError(f"Unsupported distribution name '{other}'.")

    return distribution


@find_distribution.register
def _(none: None, *args, **kwargs):
    return None


@find_distribution.register
def _(distribution: keras.Layer, *args, **kwargs):
    return distribution
