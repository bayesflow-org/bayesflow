import keras

from bayesflow.types import Tensor
from bayesflow.utils import issue_url
from bayesflow.utils.serialization import serializable

from typing import Literal

from .kernels import gaussian, inverse_multiquadratic


@serializable("bayesflow.metrics")
def maximum_mean_discrepancy(
    x: Tensor,
    y: Tensor,
    kernel: Literal["inverse_multiquadratic", "gaussian"] = "inverse_multiquadratic",
    unbiased: bool = False,
    **kwargs,
) -> Tensor:
    """Computes a mixture of Gaussian radial basis functions (RBFs) between the samples of x and y.

    See the original paper below for details and different estimators:

    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. The Journal of Machine Learning Research, 13(1), 723-773.
    https://jmlr.csail.mit.edu/papers/v13/gretton12a.html

    Parameters
    ----------
    x        :  Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y        :  Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    kernel   : str, optional (default - "inverse_multiquadratic")
        The (mixture of) kernels to be used for the MMD computation.
    unbiased : bool, optional (default - False)
        Whether to use the unbiased MMD estimator. Default is False.

    Returns
    -------
    mmd  : Tensor of shape (1, )
        The biased or unbiased empirical maximum mean discrepancy (MMD) estimator.
    """

    if kernel == "gaussian":
        kernel_fn = gaussian
    elif kernel == "inverse_multiquadratic":
        kernel_fn = inverse_multiquadratic
    else:
        raise ValueError(
            "For now, we only support a gaussian and an inverse_multiquadratic kernel."
            f"If you need a different kernel, please open an issue at {issue_url}"
        )

    if keras.ops.shape(x)[1:] != keras.ops.shape(y)[1:]:
        raise ValueError(
            f"Expected x and y to live in the same feature space, "
            f"but got {keras.ops.shape(x)[1:]} != {keras.ops.shape(y)[1:]}."
        )

    if unbiased:
        m, n = keras.ops.shape(x)[0], keras.ops.shape(y)[0]
        xx = (1.0 / (m * (m + 1))) * keras.ops.sum(kernel_fn(x, x, **kwargs))
        yy = (1.0 / (n * (n + 1))) * keras.ops.sum(kernel_fn(y, y, **kwargs))
        xy = (2.0 / (m * n)) * keras.ops.sum(kernel_fn(x, y, **kwargs))
    else:
        xx = keras.ops.mean(kernel_fn(x, x, **kwargs))
        yy = keras.ops.mean(kernel_fn(y, y, **kwargs))
        xy = keras.ops.mean(kernel_fn(x, y, **kwargs))

    return xx + yy - 2.0 * xy
