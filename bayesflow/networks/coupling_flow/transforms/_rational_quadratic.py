from typing import TypedDict

import keras

from bayesflow.types import Tensor


class Edges(TypedDict):
    left: Tensor
    right: Tensor
    bottom: Tensor
    top: Tensor


class Derivatives(TypedDict):
    left: Tensor
    right: Tensor


def _rational_quadratic_spline(
    x: Tensor, edges: Edges, derivatives: Derivatives, inverse: bool = False
) -> (Tensor, Tensor):
    # rename variables to match the paper:

    # $x^{(k)}$
    xk = edges["left"]

    # $x^{(k+1)}$
    xkp = edges["right"]

    # $y^{(k)}$
    yk = edges["bottom"]

    # $y^{(k+1)}$
    ykp = edges["top"]

    # $delta^{(k)}$
    dk = derivatives["left"]

    # $delta^{(k+1)}$
    dkp = derivatives["right"]

    # commonly used values
    dx = xkp - xk
    dy = ykp - yk
    sk = dy / dx

    if not inverse:
        xi = (x - xk) / dx

        # Eq. 4 in the paper
        numerator = dy * (sk * xi**2 + dk * xi * (1 - xi))
        denominator = sk + (dkp + dk - 2 * sk) * xi * (1 - xi)
        result = yk + numerator / denominator
    else:
        # rename for clarity
        y = x

        # Eq. 6-8 in the paper
        a = dy * (sk - dk) + (y - yk) * (dkp + dk - 2 * sk)
        b = dy * dk - (y - yk) * (dkp + dk - 2 * sk)
        c = -sk * (y - yk)

        # Eq. 29 in the appendix of the paper
        discriminant = b**2 - 4 * a * c

        # the discriminant must be positive, even when the spline is called out of bounds
        discriminant = keras.ops.maximum(discriminant, 0)

        xi = 2 * c / (-b - keras.ops.sqrt(discriminant))
        result = xi * dx + xk

    # Eq 5 in the paper
    numerator = sk**2 * (dkp * xi**2 + 2 * sk * xi * (1 - xi) + dk * (1 - xi) ** 2)
    denominator = (sk + (dkp + dk - 2 * sk) * xi * (1 - xi)) ** 2
    log_jac = keras.ops.log(numerator) - keras.ops.log(denominator)

    if inverse:
        log_jac = -log_jac

    return result, log_jac
