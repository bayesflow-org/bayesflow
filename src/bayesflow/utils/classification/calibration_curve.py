import numpy as np


def calibration_curve(
    targets: np.ndarray,
    estimates: np.ndarray,
    *,
    pos_label: int | float | bool | str = 1,
    num_bins: int = 5,
    strategy: str = "uniform",
):
    """Compute true and predicted probabilities for a calibration curve.

    The method assumes the inputs come from a binary classifier, and
    discretize the [0, 1] interval into bins.

    Code from: https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/calibration.py#L927

    Parameters
    ----------
    targets : array-like of shape (n_samples,)
        True targets.
    estimates : array-like of shape (n_samples,)
        Probabilities of the positive class.
    pos_label : int, float, bool or str, default = 1
        The label of the positive class.
    num_bins : int, default=5
        Number of bins to discretize the [0, 1] interval. A bigger number
        requires more data. Bins with no samples (i.e. without
        corresponding values in `estimates`) will not be returned, thus the
        returned arrays may have less than `num_bins` values.
    strategy : {'uniform', 'quantile'}, default='uniform'
        Strategy used to define the widths of the bins.

        uniform
            The bins have identical widths.
        quantile
            The bins have the same number of samples and depend on `y_prob`.

    Returns
    -------
    prob_true : ndarray of shape (num_bins,) or smaller
        The proportion of samples whose class is the positive class, in each
        bin (fraction of positives).

    prob_pred : ndarray of shape (num_bins,) or smaller
        The mean estimated probability in each bin.

    References
    ----------
    Alexandru Niculescu-Mizil and Rich Caruana (2005) Predicting Good
    Probabilities With Supervised Learning, in Proceedings of the 22nd
    International Conference on Machine Learning (ICML).
    """

    if estimates.min() < 0 or estimates.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(targets)
    if len(labels) > 2:
        raise ValueError(f"Only binary classification is supported. Provided labels {labels}.")
    targets = targets == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, num_bins + 1)
        bins = np.percentile(estimates, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, num_bins + 1)
    else:
        raise ValueError("Invalid entry to 'strategy' input. Strategy must be either 'quantile' or 'uniform'.")

    binids = np.searchsorted(bins[1:-1], estimates)

    bin_sums = np.bincount(binids, weights=estimates, minlength=len(bins))
    bin_true = np.bincount(binids, weights=targets, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]

    return prob_true, prob_pred
