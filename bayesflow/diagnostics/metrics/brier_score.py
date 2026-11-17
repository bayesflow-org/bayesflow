from typing import Any
from collections.abc import Sequence

import numpy as np


def brier_score(
    estimates: np.ndarray,
    targets: np.ndarray,
    model_names: Sequence[str] = None,
) -> dict[str, Any]:
    """
    Computes the (multi-class) Brier score of a model comparison network [1].

    The Brier score is a strictly proper scoring rule: it is minimized in
    expectation by the true posterior model probabilities, and thus jointly
    rewards calibration and sharpness. Lower is better, with 0 attained by
    perfectly confident, correct predictions.

    [1] Brier, G. W. (1950). Verification of forecasts expressed in terms of
    probability. Monthly Weather Review, 78(1), 1-3.

    Make sure that ``targets`` are **one-hot encoded** classes (i.e., model indices)!

    Parameters
    ----------
    estimates   : array of shape (num_sim, num_models)
        The predicted posterior model probabilities.
    targets     : array of shape (num_sim, num_models)
        The one-hot-encoded true model indices.
    model_names : Sequence[str], optional (default = None)
        Optional model names to show in the output. By default, models are called "M_" + model index.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "values" : np.ndarray
            The one-vs-rest Brier score per model, i.e., the mean squared
            difference between the predicted probability of model ``m`` and its
            one-hot indicator.
        - "aggregate" : float
            The multi-class Brier score, equal to the sum of the per-model values.
        - "metric_name" : str
            The name of the metric ("Brier Score").
        - "model_names" : list[str]
            The (inferred) model names.
    """

    if model_names is None:
        model_names = ["M_" + str(i) for i in range(1, estimates.shape[-1] + 1)]

    per_model = np.mean((estimates - targets) ** 2, axis=0)

    return dict(
        values=per_model,
        aggregate=per_model.sum(),
        metric_name="Brier Score",
        model_names=list(model_names),
    )
