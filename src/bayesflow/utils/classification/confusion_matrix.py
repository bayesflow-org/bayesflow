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

    # Normalize if required
    if normalize == "true":
        with np.errstate(all="ignore"):
            cm = cm.astype(np.float64)
            cm = np.divide(cm, cm.sum(axis=1, keepdims=True), where=cm.sum(axis=1, keepdims=True) != 0)
    elif normalize == "pred":
        with np.errstate(all="ignore"):
            cm = cm.astype(np.float64)
            cm = np.divide(cm, cm.sum(axis=0, keepdims=True), where=cm.sum(axis=0, keepdims=True) != 0)
    elif normalize == "all":
        cm = cm.astype(np.float64)
        cm /= cm.sum()

    return cm
