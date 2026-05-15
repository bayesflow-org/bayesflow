import numpy as np
from keras import ops


def model_comparison_accuracy(
    predictions: np.ndarray,
    true_models: np.ndarray,
) -> float:
    """
    Compute classification accuracy for model comparison networks.

    Works with both PMP and Bayes factor scoring rules:

    - **PMP mode** (``predictions`` shape ``(N, M)``): interprets values as
      softmax probabilities or logits; predicted model = ``argmax`` over the M
      outputs.
    - **Bayes factor mode** (``predictions`` shape ``(N, M-1)``): interprets
      values as log Bayes factors :math:`(f_1, \\ldots, f_{M-1})` relative to
      model 0 (:math:`f_0 = 0` by convention); predicted model = ``argmax``
      over :math:`(0, f_1, \\ldots, f_{M-1})`.

    The mode is inferred automatically from the shape of ``predictions``
    relative to ``true_models``.

    Parameters
    ----------
    predictions : np.ndarray of shape (N, M) or (N, M-1)
        Network outputs — either posterior model probabilities / logits (PMP
        mode) or log Bayes factors relative to model 0 (Bayes factor mode).
    true_models : np.ndarray of shape (N, M)
        One-hot encoded true model indices.

    Returns
    -------
    float
        Fraction of datasets for which the predicted model matches the true
        model.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesflow.diagnostics.metrics import model_comparison_accuracy
    >>> # PMP mode: predictions are softmax probabilities
    >>> probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.9, 0.0]])
    >>> one_hot = np.array([[1, 0, 0], [0, 1, 0]])
    >>> model_comparison_accuracy(probs, one_hot)
    1.0
    >>> # Bayes factor mode: predictions are log Bayes factors (M-1 values)
    >>> log_bfs = np.array([[-1.5, -2.0], [2.0, 0.5]])  # shape (2, 2) for M=3
    >>> model_comparison_accuracy(log_bfs, one_hot)
    1.0
    """
    predictions = ops.convert_to_numpy(predictions)
    true_models = ops.convert_to_numpy(true_models)

    num_models = true_models.shape[-1]
    true_labels = true_models.argmax(axis=-1)

    if predictions.shape[-1] == num_models - 1:
        # Bayes factor mode: prepend f_0 = 0 and take argmax
        f0 = np.zeros((predictions.shape[0], 1), dtype=predictions.dtype)
        predictions = np.concatenate([f0, predictions], axis=-1)

    predicted_labels = predictions.argmax(axis=-1)
    return float(np.mean(predicted_labels == true_labels))
