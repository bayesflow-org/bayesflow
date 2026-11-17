from typing import Sequence

import numpy as np


def confusion_matrix(targets: np.ndarray, estimates: np.ndarray, labels: Sequence = None, normalize: str = None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification or model comparison setting.

    Code inspired by: https://github.com/scikit-learn/scikit-learn/blob/98ed9dc73/sklearn/metrics/_classification.py

    Parameters
    ----------
    targets : np.ndarray
        Ground truth (correct) target values.
    estimates : np.ndarray
        Estimated targets as returned by a classifier.
    labels : Sequence, optional
        List of labels to index the matrix. This may be used to reorder or select a subset of labels.
        If None, labels that appear at least once in y_true or y_pred are used in sorted order.
    normalize : {'true', 'pred', 'all'}, optional
        Normalizes confusion matrix over the true (rows), predicted (columns)
        conditions or all the population. If None, no normalization is applied.

    Returns
    -------
    cm : np.ndarray of shape (num_labels, num_labels)
        Confusion matrix. Rows represent true classes, columns represent predicted classes.
    """

    # Get unique labels
    if labels is None:
        labels = np.unique(np.concatenate((targets, estimates)))
    else:
        labels = np.asarray(labels)

    label_to_index = {label: i for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Initialize the confusion matrix
    cm = np.zeros((num_labels, num_labels), dtype=np.int64)

    # Fill confusion matrix
    for t, p in zip(targets, estimates):
        if t in label_to_index and p in label_to_index:
            cm[label_to_index[t], label_to_index[p]] += 1

    # Normalize if required. Rows/columns whose sum is zero (a label that never
    # occurs) are left as 0 via a zero-initialized ``out`` — this also avoids the
    # "'where' used without 'out'" uninitialized-memory warning from ``np.divide``.
    if normalize == "true":
        cm = cm.astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
    elif normalize == "pred":
        cm = cm.astype(np.float64)
        col_sums = cm.sum(axis=0, keepdims=True)
        cm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0)
    elif normalize == "all":
        cm = cm.astype(np.float64)
        cm /= cm.sum()

    return cm
