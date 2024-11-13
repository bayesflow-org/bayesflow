from bayesflow.types import Tensor

from .exceptions import ShapeError


def check_lengths_same(*args):
    if len(set(map(len, args))) > 1:
        raise ValueError(f"All tuple arguments must have the same length, but lengths are {tuple(map(len, args))}.")


def check_posterior_prior_shapes(post_samples: Tensor, prior_samples: Tensor):
    """
    Checks requirements for the shapes of posterior and prior draws as
    necessitated by most diagnostic functions.

    Parameters
    ----------
    post_samples      : Tensor of shape (num_data_sets, num_post_draws, num_params)
        The posterior draws obtained from num_data_sets
    prior_samples     : Tensor of shape (num_data_sets, num_params)
        The prior draws obtained for generating num_data_sets

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    if len(post_samples.shape) != 3:
        raise ShapeError(
            "post_samples should be a 3-dimensional array, with the "
            "first dimension being the number of (simulated) data sets, "
            "the second dimension being the number of posterior draws per data set, "
            "and the third dimension being the number of parameters (marginal distributions), "
            f"but your input has dimensions {len(post_samples.shape)}"
        )
    elif len(prior_samples.shape) != 2:
        raise ShapeError(
            "prior_samples should be a 2-dimensional array, with the "
            "first dimension being the number of (simulated) data sets / prior draws "
            "and the second dimension being the number of parameters (marginal distributions), "
            f"but your input has dimensions {len(prior_samples.shape)}"
        )
    elif post_samples.shape[0] != prior_samples.shape[0]:
        raise ShapeError(
            "The number of elements over the first dimension of post_samples and prior_samples"
            f"should match, but post_samples has {post_samples.shape[0]} and prior_samples has "
            f"{prior_samples.shape[0]} elements, respectively."
        )
    elif post_samples.shape[-1] != prior_samples.shape[-1]:
        raise ShapeError(
            "The number of elements over the last dimension of post_samples and prior_samples"
            f"should match, but post_samples has {post_samples.shape[1]} and prior_samples has "
            f"{prior_samples.shape[-1]} elements, respectively."
        )
