from typing import Sequence, Mapping, Any

import numpy as np

import keras

from bayesflow.utils.exceptions import ShapeError
from bayesflow.networks import MLP


def classifier_two_sample_test(
    estimates: np.ndarray,
    targets: np.ndarray,
    metric: str = "accuracy",
    patience: int = 5,
    max_epochs: int = 1000,
    batch_size: int = 128,
    return_metric_only: bool = True,
    validation_split: float = 0.5,
    standardize: bool = True,
    mlp_widths: Sequence = (64, 64),
    **kwargs,
) -> float | Mapping[str, Any]:
    """
    C2ST metric [1] between samples from two distributions computed using a neural classifier.
    Can be computationally expensive if called in a loop[, since it needs to train the model
    for each set of samples.

    Note: works best for large numbers of samples and averaged across different posteriors.

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv:1610.06545.

    Parameters
    ----------
    estimates : np.ndarray
        Array of shape (num_samples_est, num_variables) containing samples representing estimated quantities
        (e.g., approximate posterior samples).
    targets : np.ndarray
        Array of shape (num_samples_tar, num_variables) containing target samples
        (e.g., samples from a reference posterior).
    metric : str, optional
        Metric to evaluate the classifier performance. Default is "accuracy".
    patience : int, optional
        Number of epochs with no improvement after which training will be stopped. Default is 5.
    max_epochs : int, optional
        Maximum number of epochs to train the classifier. Default is 1000.
    batch_size : int, optional
        Number of samples per batch during training. Default is 64.
    return_metric_only : bool, optional
        If True, only the final validation metric is returned. Otherwise, a dictionary with the score, classifier, and
        full training history is returned. Default is True.
    validation_split : float, optional
        Fraction of the training data to be used as validation data. Default is 0.5.
    standardize : bool, optional
        If True, both estimates and targets will be standardized using the mean and standard deviation of estimates.
        Default is True.
    mlp_widths : Sequence[int], optional
        Sequence specifying the number of units in each hidden layer of the MLP classifier. Default is (256, 256).
    **kwargs
        Additional keyword arguments. Recognized keyword:
            mlp_kwargs : dict
                Dictionary of additional parameters to pass to the MLP constructor.

    Returns
    -------
    results : float or dict
        If return_metric_only is True, returns the final validation metric (e.g., accuracy) as a float.
        Otherwise, returns a dictionary with keys "score", "classifier", and "history", where "score"
        is the final validation metric, "classifier" is the trained Keras model, and "history" contains the
        full training history.
    """

    # Error, if targets dim does not match estimates dim
    num_dims = estimates.shape[1]
    if not num_dims == targets.shape[1]:
        raise ShapeError(
            f"estimates and targets can have different number of samples (1st dim)"
            f"but must have the same dimensionality (2nd dim)"
            f"found: estimates shape {estimates.shape[1]}, targets shape {targets.shape[1]}"
        )

    # Standardize both estimates and targets relative to estimates mean and std
    if standardize:
        estimates_mean = np.mean(estimates, axis=0)
        estimates_std = np.std(estimates, axis=0)
        estimates = (estimates - estimates_mean) / estimates_std
        targets = (targets - estimates_mean) / estimates_std

    # Create data for classification task
    data = np.r_[estimates, targets]
    labels = np.r_[np.zeros((estimates.shape[0],)), np.ones((targets.shape[0],))]

    # Important: needed, since keras does not shuffle before selecting validation split
    shuffle_idx = np.random.permutation(data.shape[0])
    data = data[shuffle_idx]
    labels = labels[shuffle_idx]

    # Create and train classifier with optional stopping
    classifier = keras.Sequential(
        [MLP(widths=mlp_widths, **kwargs.get("mlp_kwargs", {})), keras.layers.Dense(1, activation="sigmoid")]
    )

    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=[metric])

    early_stopping = keras.callbacks.EarlyStopping(
        monitor=f"val_{metric}", patience=patience, restore_best_weights=True
    )

    # For now, we need to enable grads, since we turn them off by default
    if keras.backend.backend() == "torch":
        import torch

        with torch.enable_grad():
            history = classifier.fit(
                x=data,
                y=labels,
                epochs=max_epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping],
                validation_split=validation_split,
            )
    else:
        history = classifier.fit(
            x=data,
            y=labels,
            epochs=max_epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping],
            validation_split=validation_split,
        )

    if return_metric_only:
        return history.history[f"val_{metric}"][-1]
    return {"score": history.history[f"val_{metric}"][-1], "classifier": classifier, "history": history.history}
