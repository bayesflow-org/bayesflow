from keras import ops

from bayesflow.types import Tensor

# hard coded from ops.logspace(-6, 6, 11)
# to avoid pytorch errors/warnings if you want to use MPS
default_scales = ops.convert_to_tensor(
    [
        1.0000e-06,
        1.5849e-05,
        2.5119e-04,
        3.9811e-03,
        6.3096e-02,
        1.0000e00,
        1.5849e01,
        2.5119e02,
        3.9811e03,
        6.3096e04,
        1.0000e06,
    ]
)


def gaussian(x: Tensor, y: Tensor, scales: Tensor = default_scales) -> Tensor:
    """Computes a mixture of Gaussian radial basis functions (RBFs) between the samples of x and y.

    Parameters
    ----------
    x       :  Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    scales  : Tensor, optional (default - default_scales)
        List which denotes the widths of each of the Gaussians in the mixture.

    Returns
    -------
    kernel_matrix : Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x ~ P` and `y ~ Q`.
    """
    beta = 1.0 / (2.0 * scales[..., None])
    dist = x[..., None] - ops.transpose(y)
    dist = ops.transpose(ops.norm(dist, ord=2, axis=1))
    s = ops.matmul(beta, ops.reshape(dist, newshape=(1, -1)))
    return ops.reshape(ops.sum(ops.exp(-s), axis=0), newshape=ops.shape(dist))


def inverse_multiquadratic(x: Tensor, y: Tensor, scales: Tensor = default_scales) -> Tensor:
    """Computes a mixture of inverse multiquadratic RBFs between the samples of x and y.

    Parameters
    ----------
    x       :  Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    scales  : Tensor, optional (default - default_scales)
        List which denotes multiple scales for the IM-RBF kernel mixture.

    Returns
    -------
    kernel_matrix : Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x ~ P` and `y ~ Q`.
    """
    dist = ops.expand_dims(ops.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = ops.expand_dims(scales, axis=0)
    return ops.sum(sigmas / (dist + sigmas), axis=-1)
