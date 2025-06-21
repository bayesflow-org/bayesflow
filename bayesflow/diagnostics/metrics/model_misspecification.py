"""
This module provides functions for computing distances between observation samples and reference samples with distance
distributions within the reference samples for hypothesis testing.
"""

from collections.abc import Mapping, Callable

import numpy as np
from keras.ops import convert_to_numpy, convert_to_tensor

from bayesflow.approximators import ContinuousApproximator
from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor


def bootstrap_comparison(
    observed_samples: np.ndarray,
    reference_samples: np.ndarray,
    comparison_fn: Callable[[Tensor, Tensor], Tensor],
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the distance between observed and reference samples and generates a distribution of null sample
    distances by bootstrapping for hypothesis testing.

    Parameters
    ----------
    observed_samples : np.ndarray)
        Observed samples, shape (num_observed, ...).
    reference_samples : np.ndarray
        Reference samples, shape (num_reference, ...).
    comparison_fn : Callable[[Tensor, Tensor], Tensor]
        Function to compute the distance metric.
    num_null_samples : int
        Number of null samples to generate for hypothesis testing. Default is 100.

    Returns
    -------
    distance_observed : float
        The distance value between observed and reference samples.
    distance_null : np.ndarray
        A distribution of distance values under the null hypothesis.

    Raises
    ------
    ValueError
        - If the number of number of observed samples exceeds the number of reference samples
        - If the shapes of observed and reference samples do not match on dimensions besides the first one.
    """
    num_observed: int = observed_samples.shape[0]
    num_reference: int = reference_samples.shape[0]

    if num_observed > num_reference:
        raise ValueError(
            f"Number of observed samples ({num_observed}) cannot exceed"
            f"the number of reference samples ({num_reference}) for bootstrapping."
        )
    if observed_samples.shape[1:] != reference_samples.shape[1:]:
        raise ValueError(
            f"Expected observed and reference samples to have the same shape, "
            f"but got {observed_samples.shape[1:]} != {reference_samples.shape[1:]}."
        )

    observed_samples_tensor: Tensor = convert_to_tensor(observed_samples, dtype="float32")
    reference_samples_tensor: Tensor = convert_to_tensor(reference_samples, dtype="float32")

    distance_null_samples: np.ndarray = np.zeros(num_null_samples, dtype=np.float64)
    for i in range(num_null_samples):
        bootstrap_idx: np.ndarray = np.random.randint(0, num_reference, size=num_observed)
        bootstrap_samples: np.ndarray = reference_samples[bootstrap_idx]
        bootstrap_samples_tensor: Tensor = convert_to_tensor(bootstrap_samples, dtype="float32")
        distance_null_samples[i] = convert_to_numpy(comparison_fn(bootstrap_samples_tensor, reference_samples_tensor))

    distance_observed_tensor: Tensor = comparison_fn(
        observed_samples_tensor,
        reference_samples_tensor,
    )

    distance_observed: float = float(convert_to_numpy(distance_observed_tensor))

    return distance_observed, distance_null_samples


def summary_space_comparison(
    observed_data: Mapping[str, np.ndarray],
    reference_data: Mapping[str, np.ndarray],
    approximator: ContinuousApproximator,
    num_null_samples: int = 100,
    comparison_fn: Callable = maximum_mean_discrepancy,
    **kwargs,
) -> tuple[float, np.ndarray]:
    """Computes the distance between observed and reference data in the summary space and
    generates a distribution of distance values under the null hypothesis to assess model misspecification.

    By default, the Maximum Mean Discrepancy (MMD) is used as a distance function.

    [1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
    inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866.
    URL: https://arxiv.org/abs/2112.08866

    Parameters
    ----------
    observed_data : dict[str, np.ndarray]
        Dictionary of observed data as NumPy arrays, which will be preprocessed by the approximators adapter and passed
        through its summary network.
    reference_data : dict[str, np.ndarray]
        Dictionary of reference data as NumPy arrays, which will be preprocessed by the approximators adapter and passed
        through its summary network.
    approximator : ContinuousApproximator
        An instance of :py:class:`~bayesflow.approximators.ContinuousApproximator` used to compute summary statistics
        from the data.
    num_null_samples : int, optional
        Number of null samples to generate for hypothesis testing. Default is 100.
    comparison_fn : Callable, optional
        Distance function to compare the data in the summary space.
    **kwargs : dict
        Additional keyword arguments for the adapter and sampling process.

    Returns
    -------
    distance_observed : float
        The MMD value between observed and reference summaries.
    distance_null : np.ndarray
        A distribution of MMD values under the null hypothesis.

    Raises
    ------
    ValueError
        If approximator is not an instance of ContinuousApproximator or does not have a summary network.
    """

    if not isinstance(approximator, ContinuousApproximator):
        raise ValueError("The approximator must be an instance of ContinuousApproximator.")

    if not hasattr(approximator, "summary_network") or approximator.summary_network is None:
        comparison_fn_name = (
            "bayesflow.metrics.functional.maximum_mean_discrepancy"
            if comparison_fn is maximum_mean_discrepancy
            else comparison_fn.__name__
        )
        raise ValueError(
            "The approximator must have a summary network. If you have manually crafted summary "
            "statistics, or want to compare raw data and not summary statistics, please use the "
            f"`bootstrap_comparison` function with `comparison_fn={comparison_fn_name}` on the respective arrays."
        )
    observed_summaries = convert_to_numpy(approximator.summarize(observed_data))
    reference_summaries = convert_to_numpy(approximator.summarize(reference_data))

    distance_observed, distance_null = bootstrap_comparison(
        observed_samples=observed_summaries,
        reference_samples=reference_summaries,
        comparison_fn=comparison_fn,
        num_null_samples=num_null_samples,
    )

    return distance_observed, distance_null
