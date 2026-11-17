from typing import Any
from collections.abc import Sequence

import numpy as np

from ...utils.classification import confusion_matrix


def accuracy(
    predictions: np.ndarray,
    true_models: np.ndarray,
    model_names: Sequence[str] | None = None,
    per_model: bool = True,
) -> dict[str, Any]:
    """
    Compute classification accuracy for model comparison networks.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M)
        Predicted (log) probabilities. The argmax is used as the predicted label.
    true_models : np.ndarray of shape (N, M)
        One-hot encoded true model indices.
    model_names : Sequence[str], optional (default = None)
        Optional model names to show in the output. Only used when ``per_model``
        is True. By default, models are called "M_" + model index.
    per_model : bool, optional (default = True)
        If True, returns the per-model accuracy, i.e., the diagonal entries of
        the confusion matrix normalized over the true (row) labels. This is
        equivalent to the recall / true-positive-rate of each model. If False,
        returns the overall (aggregate) accuracy as a single float.

    Returns
    -------
    dict[str, any] or float
        If ``per_model`` is True, a dictionary containing:

        - "values" : np.ndarray
            The per-model accuracy (diagonal of the row-normalized confusion
            matrix).
        - "metric_name" : str
            The name of the metric ("Accuracy").
        - "model_names" : list[str]
            The (inferred) model names.

        If ``per_model`` is False, the fraction of datasets for which the
        predicted model matches the true model (a single float).

    Examples
    --------
    >>> import numpy as np
    >>> from bayesflow.diagnostics.metrics import accuracy
    >>> probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.0]])
    >>> one_hot = np.array([[1, 0, 0], [0, 1, 0]])
    >>> accuracy(probs, one_hot, per_model=False)
    1.0
    """

    num_models = true_models.shape[-1]
    true_labels = true_models.argmax(axis=-1)

    predicted_labels = predictions.argmax(axis=-1)

    if not per_model:
        return np.mean(predicted_labels == true_labels)

    if model_names is None:
        model_names = ["M_" + str(i) for i in range(1, num_models + 1)]

    cm = confusion_matrix(true_labels, predicted_labels, labels=np.arange(num_models), normalize="true")
    values = np.diag(cm)

    return dict(values=values, metric_name="Accuracy", model_names=list(model_names))
