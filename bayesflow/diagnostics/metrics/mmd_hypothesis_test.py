"""
This module provides functions for performing hypothesis testing using the Maximum Mean Discrepancy (MMD) metric.

[1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866., https://arxiv.org/abs/2112.08866

Functions:
----------
- compute_mmd_hypothesis_test_from_summaries:
    Computes the MMD between observed and reference summaries and generates a null distribution of MMD values
    for hypothesis testing.

- compute_mmd_hypothesis_test:
    Computes the MMD between observed and reference data using an approximator to extract summary statistics,
    and generates a null distribution of MMD values for hypothesis testing.

Dependencies:
-------------
- numpy: For numerical operations.
- bayesflow.approximators: Provides the `Approximator` class for extracting summary statistics.
- bayesflow.metrics: Provides the `maximum_mean_discrepancy` function for computing the MMD.
"""

import numpy as np
from keras.ops import convert_to_numpy, convert_to_tensor

from bayesflow.approximators import ContinuousApproximator
from bayesflow.networks import SummaryNetwork
from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor


def compute_mmd_hypothesis_test_from_summaries(
    observed_summaries: np.ndarray,
    reference_summaries: np.ndarray,
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference summaries and generates a distribution
    of MMD values under the null hypothesis to assess model fit.

    [1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
    inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866.
    URL: https://arxiv.org/abs/2112.08866


    Example:
    --------
    # Assuming `observed_summaries` and `reference_summaries` are available:

    from bayesflow.diagnostics.metrics import compute_mmd_hypothesis_test_from_summaries
    from bayesflow.diagnostics.plots import mmd_hypothesis_test

    # Compute MMD values for hypothesis test
    mmd_observed, mmd_null = compute_mmd_hypothesis_test_from_summaries(observed_summaries, reference_summaries)

    # Plot the null distribution and observed MMD
    fig = mmd_hypothesis_test(mmd_null=mmd_null, mmd_observed=mmd_observed)


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

    Raises
    ------
    ValueError
        - If number of reference summaries is less than number of observed summaries.
        - If the shapes of observed and reference summaries do not match on dimensions besides the first one.
        - If number of null samples is less than or equal to 0.
    """
    num_observed: int = observed_summaries.shape[0]
    num_reference: int = reference_summaries.shape[0]

    if num_observed > num_reference:
        raise ValueError(
            f"Number of reference summaries ({num_reference}) must be greater than"
            f"number of observed summaries ({num_observed})."
        )

    if observed_summaries.shape[1:] != reference_summaries.shape[1:]:
        raise ValueError(
            f"Expected observed and reference summaries to have the same shape, "
            f"but got {observed_summaries.shape[1:]} != {reference_summaries.shape[1:]}."
        )

    if num_null_samples <= 0:
        raise ValueError(f"Number of null samples must be greater than 0, but got {num_null_samples}.")

    observed_summaries_tensor: Tensor = convert_to_tensor(observed_summaries, dtype="float32")
    reference_summaries_tensor: Tensor = convert_to_tensor(reference_summaries, dtype="float32")

    mmd_null_samples: np.ndarray = np.zeros(num_null_samples, dtype=np.float64)

    for i in range(num_null_samples):
        bootstrap_idx: np.ndarray = np.random.randint(0, num_reference, size=num_observed)
        sampled_summaries: np.ndarray = reference_summaries[bootstrap_idx]
        sampled_summaries_tensor: Tensor = convert_to_tensor(sampled_summaries, dtype="float32")
        mmd_null_samples[i] = convert_to_numpy(
            maximum_mean_discrepancy(
                sampled_summaries_tensor,
                reference_summaries_tensor,
            )
        )

    mmd_observed_tensor: Tensor = maximum_mean_discrepancy(
        observed_summaries_tensor,
        reference_summaries_tensor,
    )

    mmd_observed: float = float(convert_to_numpy(mmd_observed_tensor))

    return mmd_observed, mmd_null_samples


def compute_mmd_hypothesis_test(
    observed_data: np.ndarray,
    reference_data: np.ndarray,
    approximator: ContinuousApproximator | SummaryNetwork,
    num_null_samples: int = 100,
) -> tuple[float, np.ndarray]:
    """Computes the Maximum Mean Discrepancy (MMD) between observed and reference data and generates a distribution of
    MMD values under the null hypothesis to assess model fit.

    [1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
    inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866.
    URL: https://arxiv.org/abs/2112.08866


    Example:
    --------
    # Assuming `observed_data`, `reference_data`, and an `Approximator` instance are available:

    from bayesflow.diagnostics.metrics import compute_mmd_hypothesis_test
    from bayesflow.diagnostics.plots import mmd_hypothesis_test

    # Compute MMD values for hypothesis test
    mmd_observed, mmd_null = compute_mmd_hypothesis_test(observed_data, reference_data, approximator)

    # Plot the null distribution and observed MMD
    fig = mmd_hypothesis_test(mmd_null=mmd_null, mmd_observed=mmd_observed)


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

    mmd_observed, mmd_null = compute_mmd_hypothesis_test_from_summaries(
        observed_summaries, reference_summaries, num_null_samples=num_null_samples
    )

    return mmd_observed, mmd_null
