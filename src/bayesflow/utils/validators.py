from bayesflow.types import Tensor

from .exceptions import ShapeError


def check_lengths_same(*args):
    if len(set(map(len, args))) > 1:
        raise ValueError(f"All tuple arguments must have the same length, but lengths are {tuple(map(len, args))}.")


def check_prior_shapes(variables: Tensor):
    """
    Checks the shape of posterior draws as required by most diagnostic functions

    Parameters
    ----------
    variables     : Tensor of shape (num_data_sets, num_params)
        The prior_samples from generating num_data_sets
    """

    if len(variables.shape) != 2:
        raise ShapeError(
            "prior_samples samples should be a 2-dimensional array, with the "
            "first dimension being the number of (simulated) data sets / prior_samples draws "
            "and the second dimension being the number of variables, "
            f"but your input has dimensions {len(variables.shape)}"
        )


def check_estimates_shapes(variables: Tensor):
    """
    Checks the shape of model-generated predictions (posterior draws, point estimates)
    as required by most diagnostic functions

    Parameters
    ----------
    variables     : Tensor of shape (num_data_sets, num_post_draws, num_params)
        The prior_samples from generating num_data_sets
    """
    if len(variables.shape) != 2 and len(variables.shape) != 3:
        raise ShapeError(
            "estimates should be a 2- or 3-dimensional array, with the "
            "first dimension being the number of data sets, "
            "(optional) second dimension the number of posterior draws per data set, "
            "and the last dimension the number of estimated variables, "
            f"but your input has dimensions {len(variables.shape)}"
        )


def check_consistent_shapes(estimates: Tensor, prior_samples: Tensor):
    """
    Checks whether the model-generated predictions (posterior draws, point estimates) and
    prior_samples have consistent leading (num_data_sets) and trailing (num_params) dimensions
    """
    if estimates.shape[0] != prior_samples.shape[0]:
        raise ShapeError(
            "The number of elements over the first dimension of estimates and prior_samples"
            f"should match, but estimates have {estimates.shape[0]} and prior_samples has "
            f"{prior_samples.shape[0]} elements, respectively."
        )
    if estimates.shape[-1] != prior_samples.shape[-1]:
        raise ShapeError(
            "The number of elements over the last dimension of estimates and prior_samples"
            f"should match, but estimates has {estimates.shape[0]} and prior_samples has "
            f"{prior_samples.shape[0]} elements, respectively."
        )


def check_estimates_prior_shapes(estimates: Tensor, prior_samples: Tensor):
    """
    Checks requirements for the shapes of estimates and prior_samples draws as
    necessitated by most diagnostic functions.

    Parameters
    ----------
    estimates      : Tensor of shape (num_data_sets, num_post_draws, num_params) or (num_data_sets, num_params)
        The model-generated predictions (posterior draws, point estimates) obtained from num_data_sets
    prior_samples     : Tensor of shape (num_data_sets, num_params)
        The prior_samples draws obtained for generating num_data_sets

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `estimates` and `estimates`.
    """

    check_estimates_shapes(estimates)
    check_prior_shapes(prior_samples)
    check_consistent_shapes(estimates, prior_samples)
