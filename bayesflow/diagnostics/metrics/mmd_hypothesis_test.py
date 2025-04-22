"""
This module provides functions for computing distances between observation samples and reference samples with distance
distributions within the reference samples for hypothesis testing.

Functions:
----------
- bootstrap_comparison: Computes distance between observed and reference samples and generates a distribution of null
  sample distances by bootstrapping for hypothesis testing.
- mmd_comparison_from_summaries: Computes the Maximum Mean Discrepancy (MMD) between observed and reference summaries
  and generates a distribution of MMD values under the null hypothesis to assess model misspecification.
- mmd_comparison: Computes the Maximum Mean Discrepancy (MMD) between observed and reference data and generates a
  distribution of MMD values under the null hypothesis to assess model misspecification.

Dependencies:
-------------
- numpy: For numerical operations.
- keras.ops: For converting data to numpy and tensor formats.
- bayesflow.networks: Provides the `SummaryNetwork` class for extracting summary statistics.
- bayesflow.approximators: Provides the `Approximator` class for extracting summary statistics.
- bayesflow.metrics: Provides the `maximum_mean_discrepancy` function for computing the MMD.
"""

import typing

import numpy as np
from keras.ops import convert_to_numpy, convert_to_tensor

from bayesflow.approximators import ContinuousApproximator
from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.networks import SummaryNetwork
from bayesflow.types import Tensor


def bootstrap_comparison(
    observed_samples: np.ndarray,
    reference_samples: np.ndarray,
    comparison_fn: typing.Callable[[Tensor, Tensor], Tensor],
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Compute distance between observed and reference samples and generated a distribution of null sample distances by
    bootstrapping for hypothesis testing.

    Parameters
    ----------
    observed_samples : np.ndarray)
        Observed samples, shape (num_observed, ...).
    reference_samples : np.ndarray
        Reference samples, shape (num_reference, ...).
    comparison_fn : typing.Callable[[Tensor, Tensor], Tensor]
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


def mmd_comparison_from_summaries(
    observed_summaries: np.ndarray,
    reference_summaries: np.ndarray,
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference summaries and generates a distribution
    of MMD values under the null hypothesis to assess model misspecification.

    [1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
    inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866.
    URL: https://arxiv.org/abs/2112.08866


    Parameters
    ----------
    observed_summary : np.ndarray
        Summary statistics of observed data, shape (num_observed, ...).
    reference_summary : np.ndarray
        Summary statistics of reference data, shape (num_reference, ...).
    num_null_samples : int
        Number of null samples to generate for hypothesis testing. Default is 100.

    Returns
    -------
    mmd_observed : float
        The MMD value between observed and reference summaries.
    mmd_null : np.ndarray
        A distribution of MMD values under the null hypothesis.
    """
    mmd_observed, mmd_null_samples = bootstrap_comparison(
        observed_samples=observed_summaries,
        reference_samples=reference_summaries,
        comparison_fn=maximum_mean_discrepancy,
        num_null_samples=num_null_samples,
    )

    return mmd_observed, mmd_null_samples


def mmd_comparison(
    observed_data: np.ndarray,
    reference_data: np.ndarray,
    approximator: ContinuousApproximator | SummaryNetwork,
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference data and generates a distribution of
    MMD values under the null hypothesis to assess model misspecification.

    [1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
    inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866.
    URL: https://arxiv.org/abs/2112.08866


    Parameters
    ----------
    observed_data : np.ndarray
        Observed data, shape (num_observed, ...).
    reference_data : np.ndarray
        Reference data, shape (num_reference, ...).
    approximator : ContinuousApproximator or SummaryNetwork
        An instance of the ContinuousApproximator or SummaryNetwork class use to extract summary statistics from data.
    num_null_samples : int
        Number of null samples to generate for hypothesis testing. Default is 100.

    Returns
    -------
    mmd_observed : float
        The MMD value between observed and reference data.
    mmd_null : np.ndarray
        A distribution of MMD values under the null hypothesis.

    Raises:
    ------
    ValueError
        - If the shapes of observed and reference data do not match on dimensions besides the first one.
        - If approximator is not an instance of ContinuousApproximator or SummaryNetwork.
    """
    if observed_data.shape[1:] != reference_data.shape[1:]:
        raise ValueError(
            f"Expected observed and reference data to have the same shape, "
            f"but got {observed_data.shape[1:]} != {reference_data.shape[1:]}."
        )

    if isinstance(approximator, ContinuousApproximator):
        if approximator.summary_network is not None:
            observed_data_tensor: Tensor = convert_to_tensor(observed_data)
            reference_data_tensor: Tensor = convert_to_tensor(reference_data)
            observed_summaries: np.ndarray = convert_to_numpy(approximator.summary_network(observed_data_tensor))
            reference_summaries: np.ndarray = convert_to_numpy(approximator.summary_network(reference_data_tensor))
        else:
            observed_summaries: np.ndarray = observed_data
            reference_summaries: np.ndarray = reference_data
    elif isinstance(approximator, SummaryNetwork):
        observed_data_tensor: Tensor = convert_to_tensor(observed_data)
        reference_data_tensor: Tensor = convert_to_tensor(reference_data)
        observed_summaries: np.ndarray = convert_to_numpy(approximator(observed_data_tensor))
        reference_summaries: np.ndarray = convert_to_numpy(approximator(reference_data_tensor))
    else:
        raise ValueError("The approximator must be an instance of ContinuousApproximator or SummaryNetwork.")

    mmd_observed, mmd_null = mmd_comparison_from_summaries(
        observed_summaries, reference_summaries, num_null_samples=num_null_samples
    )

    return mmd_observed, mmd_null
