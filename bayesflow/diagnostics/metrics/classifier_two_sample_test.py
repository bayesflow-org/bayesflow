from typing import Sequence, Mapping, Any, Callable, Literal

import numpy as np

import keras
from scipy.special import expit
from scipy.stats import norm, rankdata

from bayesflow.utils import logging
from bayesflow.utils.exceptions import ShapeError
from bayesflow.networks import MLP


def classifier_two_sample_test(
    estimates: np.ndarray,
    targets: np.ndarray,
    metric: str = "accuracy",
    patience: int = 5,
    min_epochs: int = 0,
    max_epochs: int = 1000,
    batch_size: int = 128,
    return_metric_only: bool = True,
    cross_validation_splits: int = 5,
    validation_split: float = 0.5,
    early_stopping_split: float = 0.2,
    standardize: bool = True,
    mlp_widths: Sequence | Literal["auto"] = "auto",
    classifier: keras.Model | Callable[[], keras.Model] | None = None,
    conformal: bool | None = None,
    num_permutations: int = 0,
    seed: int | None = None,
    **kwargs,
) -> float | Mapping[str, Any]:
    """
    C2ST metric [1, 4] between samples from two distributions, computed with a neural classifier.
    Can be expensive in a loop, since each call trains at least one classifier.

    Besides the classification metric, two statistics are computed from the held-out classifier scores:
    the global regression statistic ``mean((m(x) - pi)^2)`` of [2], which is only interpretable relative
    to its permutation null since classifier variance inflates it, and the AUC, which is rank-based and
    therefore robust to weak, miscalibrated or overfit classifiers [3]. P-values are available from a
    label permutation test (``num_permutations``) and, for the AUC, from the conformal test of [3]
    (``conformal``).

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv:1610.06545.

    [2] Kim, I., Lee, A. B., & Lei, J. (2019). Global and local two-sample tests via regression.
    arXiv:1812.08927.

    [3] Bansal, V., Chen, T., & Scott, J. G. (2026). Conformal C2ST: Turning weak classifiers into strong
    two-sample tests. ICML 2026. arXiv:2507.17026.

    [4] Yao, Y., & Domke, J. (2023). Discriminative calibration: Check Bayesian computation from simulations
    and flexible classifier. NeurIPS 2023. arXiv:2305.14593.

    Parameters
    ----------
    estimates : np.ndarray
        Array of shape (num_samples_est, num_variables), e.g., approximate posterior samples.
    targets : np.ndarray
        Array of shape (num_samples_tar, num_variables), e.g., samples from a reference posterior.
    metric : str
        Classifier metric in [0, 1] where larger is better; mapped to >= 0.5. Default is "accuracy".
    patience : int
        Number of epochs without improvement after which training stops. Default is 5.
    min_epochs : int
        Number of warm-up epochs during which early stopping is disabled. Default is 0.
    max_epochs : int
        Maximum number of epochs to train the classifier. Default is 1000.
    batch_size : int
        Number of samples per batch during training. Default is 128.
    return_metric_only : bool
        If True, only the validation metric is returned; otherwise, also the other statistics, classifiers
        and histories. Ignored if ``conformal`` is True or ``num_permutations`` is set. Default is True.
    cross_validation_splits : int
        Number of cross-validation splits. Default is 5.
    validation_split : float
        Fraction of the data used as validation data for a single hold-out split
        (``cross_validation_splits=1``). Default is 0.5.
    early_stopping_split : float
        Fraction of the training data held out for early stopping. Set to 0 to disable early
        stopping. Default is 0.2.
    standardize : bool
        If True, the pooled samples are standardized. Default is True.
    mlp_widths : Sequence[int] or "auto"
        Hidden layer widths of the default MLP. 'auto' uses two layers of the smallest power of two above
        10 times the number of variables. Ignored if ``classifier`` is passed. Default is 'auto'.
    classifier : keras.Model or callable, optional
        Classifier to use instead of the default MLP: a Keras model (cloned) or a callable returning one.
        It must map (num_samples, num_variables) to a probability in [0, 1]; uncompiled models are compiled
        with Adam, binary cross-entropy and `metric``. If it ends in a dense sigmoid unit (like the default MLP),
        its pre-sigmoid log-odds are used as scores for the rank-based statistics. Default is None.
    conformal : bool, optional
        Whether to additionally compute the conformal two-sample test of [3] (their budget-matched
        "multiple" variant), which calibrates the ranks of the held-out scores without re-training and
        therefore stays powerful for weak, biased or overfit classifiers. Default is None:
        the test is included whenever a dictionary is returned anyway, since it costs no extra fits.
    num_permutations : int
        Number of label permutations for permutation p-values of the classification metric, the regression
        statistic and the AUC, with resolution ``1 / (num_permutations + 1)``. Default is 0 (no test).
    seed : int, optional
        Seed for reproduciblity. Default is None (non-deterministic).
    **kwargs
        Additional keyword arguments. Recognized keyword:
            mlp_kwargs : dict
                Dictionary of additional parameters to pass to the MLP constructor.

    Returns
    -------
    results : float or dict
    """
    num_dims = estimates.shape[1]
    if not num_dims == targets.shape[1]:
        raise ShapeError(
            f"estimates and targets can have a different number of samples (1st dim), "
            f"but must have the same dimensionality (2nd dim). "
            f"Found: estimates shape {estimates.shape[1]}, targets shape {targets.shape[1]}"
        )

    num_estimates, num_targets = estimates.shape[0], targets.shape[0]
    if num_estimates != num_targets:
        logging.warning(
            "Found {} estimates but {} targets. The classifier is trained with class weights to compensate, "
            "but the classification metric is still dominated by the ratio of the two sample sizes. Prefer "
            "the AUC and the conformal test, which account for unequal sample sizes.",
            num_estimates,
            num_targets,
        )

    # Include the conformal test whenever a dictionary of statistics is returned anyway
    if conformal is None:
        conformal = not return_metric_only or num_permutations > 0

    rng = np.random.default_rng(seed)
    if seed is not None:
        # The weight initialization and batch shuffling of the classifier are not controlled by rng
        keras.utils.set_random_seed(seed)

    if mlp_widths == "auto":
        widths = 2 ** int(np.ceil(np.log2(10 * num_dims)))
        mlp_widths = [widths, widths]

    data = np.r_[estimates, targets]
    labels = np.r_[np.zeros((num_estimates,)), np.ones((num_targets,))]

    # Standardize the pooled samples
    if standardize:
        data_std = np.std(data, axis=0)
        data = (data - np.mean(data, axis=0)) / np.where(data_std == 0, 1.0, data_std)

    def build_classifier():
        """Build a freshly initialized classifier."""
        if classifier is None:
            model = keras.Sequential(
                [
                    MLP(widths=mlp_widths, **kwargs.get("mlp_kwargs", {})),
                    keras.layers.Dense(units=1, activation="sigmoid"),
                ]
            )
        elif isinstance(classifier, keras.Model):
            # Cloning keeps the compile configuration, but re-initializes the weights
            model = keras.models.clone_model(classifier)
        elif callable(classifier):
            model = classifier()
        else:
            raise TypeError(f"classifier must be a keras.Model or a callable returning one, but found: {classifier}")

        if not model.compiled:
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[metric])
        return model

    def fit_splits(split_labels: np.ndarray) -> list[dict]:
        """Train one classifier per split and collect its held-out scores."""
        folds = []
        for train_idx, val_idx in _make_splits(split_labels, cross_validation_splits, validation_split, rng):
            data_train, data_val = data[train_idx], data[val_idx]
            labels_train, labels_val = split_labels[train_idx], split_labels[val_idx]

            # Weight the loss inversely to the class sizes, so that unequal sample sizes do not bias
            counts = np.bincount(labels_train.astype(int), minlength=2)
            class_weight = {label: len(labels_train) / (2 * max(count, 1)) for label, count in enumerate(counts)}

            classifier = build_classifier()
            callbacks = []
            if early_stopping_split > 0:
                callbacks.append(
                    keras.callbacks.EarlyStopping(
                        monitor=f"val_{metric}",
                        patience=patience,
                        restore_best_weights=True,
                        start_from_epoch=min_epochs,
                    )
                )
            fit_kwargs = dict(
                x=data_train,
                y=labels_train,
                epochs=max_epochs,
                batch_size=batch_size,
                verbose=0,
                callbacks=callbacks,
                validation_split=early_stopping_split,
                class_weight=class_weight,
            )

            # Gradients are disabled by default, so they need to be enabled for training
            if keras.backend.backend() == "torch":
                import torch

                with torch.enable_grad():
                    history = classifier.fit(**fit_kwargs)
            else:
                history = classifier.fit(**fit_kwargs)

            # Metric and classifier scores on the held-out fold
            evaluation = classifier.evaluate(data_val, labels_val, verbose=0, return_dict=True)
            if metric not in evaluation:
                raise ValueError(
                    f"The metric '{metric}' was not found in the evaluation results {sorted(evaluation)}. "
                    f"Please pass a metric that Keras exposes under this name."
                )
            score = evaluation[metric]
            probabilities, ranking_scores = _predict_scores(classifier, data_val)
            folds.append(
                {
                    "score": float(score),
                    "predictions": probabilities,
                    "ranking_scores": ranking_scores,
                    "labels": labels_val,
                    "history": history.history,
                    "classifier": classifier,
                }
            )
        return folds

    def summarize(folds: list[dict]) -> dict:
        """Aggregate the test statistics of all splits."""
        scores = np.array([fold["score"] for fold in folds])
        mean_score = float(np.mean(scores))
        return {
            "score": max(mean_score, 1 - mean_score),
            "scores": scores,
            "regression_statistic": float(
                np.mean([_regression_statistic(fold["predictions"], fold["labels"]) for fold in folds])
            ),
            "auc": float(np.mean([_auc(*_split_by_label(fold)) for fold in folds])),
        }

    folds = fit_splits(labels)
    results = summarize(folds)

    if conformal:
        statistics, p_values, held_out_sizes = [], [], []
        for fold in folds:
            target_scores, estimate_scores = _split_by_label(fold)
            statistic, p_value = _conformal_test(target_scores, estimate_scores, rng)
            statistics.append(statistic)
            p_values.append(p_value)
            held_out_sizes.append(min(len(target_scores), len(estimate_scores)))

        if min(held_out_sizes) < 30:
            logging.warning(
                "The conformal test is only asymptotically valid and its normal approximation needs many "
                "held-out samples from both distributions, but merely {} samples of one distribution are "
                "held out per split. Consider passing more samples or reducing cross_validation_splits.",
                min(held_out_sizes),
            )

        # Combine the p-values of the splits
        if len(p_values) > 1:
            combined_p_value = min(1.0, 2 * float(np.mean(p_values)))
        else:
            combined_p_value = p_values[0]

        results |= {
            "conformal_statistic": float(np.mean(statistics)),
            "conformal_statistics": statistics,
            "conformal_p_value": combined_p_value,
            "conformal_p_values": p_values,
        }

    if num_permutations > 0:
        # Statistics of interest, mapped such that larger values indicate a larger discrepancy
        test_statistics = {
            "score": lambda summary: summary["score"],
            "regression_statistic": lambda summary: summary["regression_statistic"],
            "auc": lambda summary: abs(summary["auc"] - 0.5),
        }

        # Under the null hypothesis, the pooled labels are exchangeable, so re-running the whole
        # procedure on permuted labels samples from the null distribution of the statistics
        observed = results
        null_summaries = (summarize(fit_splits(rng.permutation(labels))) for _ in range(num_permutations))

        null_statistics = {key: [] for key in test_statistics}
        for summary in null_summaries:
            for key, statistic in test_statistics.items():
                null_statistics[key].append(statistic(summary))

        results["permutation_p_values"] = {
            key: float((1 + np.sum(np.array(null_statistics[key]) >= statistic(observed))) / (num_permutations + 1))
            for key, statistic in test_statistics.items()
        }

    if return_metric_only and not conformal and num_permutations == 0:
        return results["score"]

    if not return_metric_only:
        results["classifiers"] = [fold["classifier"] for fold in folds]
        results["histories"] = [fold["history"] for fold in folds]

    return results


def _make_splits(
    labels: np.ndarray, cross_validation_splits: int, validation_split: float, rng: np.random.Generator
) -> list[tuple]:
    """Stratified cross-validation or hold-out splits of the pooled samples."""
    # Permute indices of each class separately to ensure stratification
    class_indices = [rng.permutation(np.where(labels == label)[0]) for label in (0, 1)]

    if cross_validation_splits > 1:
        class_folds = [np.array_split(indices, cross_validation_splits) for indices in class_indices]
        validation_indices = [
            np.concatenate([folds[i] for folds in class_folds]) for i in range(cross_validation_splits)
        ]
    else:
        num_validation = [max(1, int(round(len(indices) * validation_split))) for indices in class_indices]
        validation_indices = [np.concatenate([indices[:num] for indices, num in zip(class_indices, num_validation)])]

    splits = []
    for val_idx in validation_indices:
        mask = np.ones(len(labels), dtype=bool)
        mask[val_idx] = False
        # the training indices need to be shuffled, since keras does not shuffle before
        # selecting the validation split used for early stopping
        splits.append((rng.permutation(np.where(mask)[0]), val_idx))
    return splits


def _predict_scores(model: keras.Model, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicted probabilities and monotone ranking scores of the given samples. If the final layer is a
    dense sigmoid unit, its pre-sigmoid log-odds serve as ranking scores (Bansal et al., 2026).
    """
    last = model.layers[-1] if getattr(model, "layers", None) else None
    if isinstance(last, keras.layers.Dense) and last.units == 1 and last.activation is keras.activations.sigmoid:
        try:
            feature_model = keras.Model(model.inputs, last.input)
        except (AttributeError, ValueError):
            # The model does not expose a symbolic graph, e.g. a subclassed model
            feature_model = None
        if feature_model is not None:
            features = np.asarray(feature_model.predict(inputs, verbose=0), dtype=np.float64)
            weights = last.get_weights()
            log_odds = (features @ weights[0] + (weights[1] if len(weights) > 1 else 0.0)).reshape(-1)
            return expit(log_odds), log_odds

    probabilities = np.asarray(model.predict(inputs, verbose=0), dtype=np.float64).reshape(-1)
    return probabilities, probabilities


def _split_by_label(fold: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Held-out ranking scores of the target samples (label 1) and of the estimates (label 0)."""
    return fold["ranking_scores"][fold["labels"] == 1], fold["ranking_scores"][fold["labels"] == 0]


def _regression_statistic(scores: np.ndarray, labels: np.ndarray) -> float:
    """Global regression statistic ``mean((m(x) - pi)^2)`` of Kim et al. (2019)."""
    return float(np.mean((scores - np.mean(labels)) ** 2))


def _auc(target_scores: np.ndarray, estimate_scores: np.ndarray) -> float:
    """Area under the ROC curve ``P(s_tar > s_est) + P(s_tar = s_est) / 2`` via mid-ranks."""
    num_targets, num_estimates = len(target_scores), len(estimate_scores)
    ranks = rankdata(np.concatenate([target_scores, estimate_scores]))
    u_statistic = ranks[:num_targets].sum() - num_targets * (num_targets + 1) / 2
    return float(u_statistic / (num_targets * num_estimates))


def _conformal_test(
    target_scores: np.ndarray, estimate_scores: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """
    Budget-matched conformal ("multiple") test of Bansal et al. (2026): one conformal p-value per
    estimate against the target scores as calibration set, whose average is standardized to be
    asymptotically standard normal under the null. Returns the test statistic and its one-sided
    p-value; large statistics and small p-values indicate a difference.
    """
    num_calibration, num_test = len(target_scores), len(estimate_scores)
    if num_calibration < 2 or num_test < 2:
        raise ValueError(
            f"The conformal test needs at least 2 held-out samples per distribution, but found "
            f"{num_calibration} target and {num_test} estimate samples."
        )

    # Conformal p-value of each estimate, with uniform tie-breaking
    calibration = np.sort(target_scores)
    num_below = np.searchsorted(calibration, estimate_scores, side="left")
    num_equal = np.searchsorted(calibration, estimate_scores, side="right") - num_below
    tie_breaks = rng.uniform(size=num_test)
    conformal_p_values = (num_below + tie_breaks * num_equal) / num_calibration

    # Variance of the average p-value, which is a two-sample U-statistic: the mid-rank of each calibration
    # score within the test scores contributes its empirical variance, the test scores contribute the
    # variance 1/12 of the uniform p-values under the null.
    test_sorted = np.sort(estimate_scores)
    cdf = np.searchsorted(test_sorted, target_scores, side="right") / num_test
    cdf_left = np.searchsorted(test_sorted, target_scores, side="left") / num_test
    variance = np.var((cdf + cdf_left) / 2, ddof=1) + num_calibration / (12 * num_test)

    statistic = (0.5 - np.mean(conformal_p_values)) / np.sqrt(variance / num_calibration)
    return float(statistic), float(norm.sf(statistic))
