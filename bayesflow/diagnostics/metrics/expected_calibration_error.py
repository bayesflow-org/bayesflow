from collections.abc import Sequence

import numpy as np
from keras import ops

from ...utils.exceptions import ShapeError
from ...utils.classification import calibration_curve


def expected_calibration_error(
    estimates: np.ndarray,
    targets: np.ndarray,
    model_names: Sequence[str] = None,
    num_bins: int = 10,
    return_probs: bool = False,
) -> dict[str, any]:
    """
    Estimates the expected calibration error (ECE) of a model comparison network according to [1].

    [1] Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). Obtaining well calibrated probabilities using
    Bayesian binning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 29, No. 1).

    Notes
    -----
    Make sure that ``targets`` are **one-hot encoded** classes (i.e., model indices)!

    Parameters
    ----------
    estimates      : array of shape (num_sim, num_models)
        The predicted posterior model probabilities.
    targets      : array of shape (num_sim, num_models)
        The one-hot-encoded true model indices.
    model_names : Sequence[str], optional (default = None)
        Optional model names to show in the output. By default, models are called "M_" + model index.
    num_bins    : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).
        Passed into ``bayesflow.utils.calibration_curve()``.
    return_probs : bool (default = False)
        Do you want to obtain the output of ``bayesflow.utils.calibration_curve()``?

    Returns
    -------
    result : dict
        Dictionary containing:
        - "values" : np.ndarray
            The expected calibration error per model
        - "metric_name" : str
            The name of the metric ("Expected Calibration Error").
        - "model_names" : str
            The (inferred) variable names.
        - "probs_true": (optional) list[np.ndarray]:
            Outputs of ``bayesflow.utils.calibration.calibration_curve()`` per model
        - "probs_pred": (optional) list[np.ndarray]:
            Outputs of ``bayesflow.utils.calibration.calibration_curve()`` per model
    """

    # Convert tensors to numpy, if passed
    estimates = ops.convert_to_numpy(estimates)
    targets = ops.convert_to_numpy(targets)

    if estimates.shape != targets.shape:
        raise ShapeError("`estimates` and `targets` must have the same shape.")

    if model_names is None:
        model_names = ["M_" + str(i) for i in range(estimates.shape[-1])]
    elif len(model_names) != estimates.shape[-1]:
        raise ShapeError("There must be exactly one `model_name` for each model in `estimates`")

    # Extract number of models and prepare containers
    ece = []
    probs_true = []
    probs_pred = []

    targets = targets.argmax(axis=-1)

    # Loop for each model and compute calibration errs per bin
    for model_index in range(estimates.shape[-1]):
        y_true = (targets == model_index).astype(np.float32)
        y_prob = estimates[..., model_index]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, num_bins=num_bins)

        # Compute ECE by weighting bin errors by bin size
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        error = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

        ece.append(error)
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)

    output = dict(values=np.array(ece), metric_name="Expected Calibration Error", model_names=model_names)

    if return_probs:
        output["probs_true"] = probs_true
        output["probs_pred"] = probs_pred

    return output
