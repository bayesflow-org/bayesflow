import numpy as np
import pytest

from bayesflow.utils.classification import calibration_curve, confusion_matrix


def test_confusion_matrix_counts():
    targets = np.array([0, 0, 1, 1, 2, 2])
    estimates = np.array([0, 1, 1, 1, 2, 0])

    cm = confusion_matrix(targets, estimates)

    expected = np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert cm.dtype == np.int64
    assert np.array_equal(cm, expected)


def test_confusion_matrix_normalize_true():
    targets = np.array([0, 0, 1, 1, 2, 2])
    estimates = np.array([0, 1, 1, 1, 2, 0])

    cm = confusion_matrix(targets, estimates, normalize="true")

    assert np.allclose(cm.sum(axis=1), 1.0)
    assert np.allclose(cm[1], [0.0, 1.0, 0.0])


def test_confusion_matrix_normalize_pred():
    targets = np.array([0, 0, 1, 1, 2, 2])
    estimates = np.array([0, 1, 1, 1, 2, 0])

    cm = confusion_matrix(targets, estimates, normalize="pred")

    assert np.allclose(cm.sum(axis=0), 1.0)
    assert np.allclose(cm[:, 2], [0.0, 0.0, 1.0])


def test_confusion_matrix_normalize_all():
    targets = np.array([0, 0, 1, 1, 2, 2])
    estimates = np.array([0, 1, 1, 1, 2, 0])

    cm = confusion_matrix(targets, estimates, normalize="all")

    assert np.isclose(cm.sum(), 1.0)


def test_confusion_matrix_explicit_labels():
    targets = np.array([0, 0, 1, 1, 2, 2])
    estimates = np.array([0, 1, 1, 1, 2, 0])

    # Reordered labels permute rows/columns accordingly
    cm = confusion_matrix(targets, estimates, labels=[2, 1, 0])
    default = confusion_matrix(targets, estimates)
    assert np.array_equal(cm, default[::-1, ::-1])

    # A subset of labels restricts the matrix to those classes
    cm_subset = confusion_matrix(targets, estimates, labels=[0, 1])
    assert cm_subset.shape == (2, 2)
    assert np.array_equal(cm_subset, default[:2, :2])


def test_confusion_matrix_absent_label_normalize():
    # Label 2 never occurs: its row/column must stay all-zero under
    # normalization instead of producing NaNs
    targets = np.array([0, 0, 1, 1])
    estimates = np.array([0, 1, 1, 0])

    cm_true = confusion_matrix(targets, estimates, labels=[0, 1, 2], normalize="true")
    cm_pred = confusion_matrix(targets, estimates, labels=[0, 1, 2], normalize="pred")

    assert not np.isnan(cm_true).any()
    assert not np.isnan(cm_pred).any()
    assert np.allclose(cm_true[2], 0.0)
    assert np.allclose(cm_pred[:, 2], 0.0)


def test_calibration_curve_strategies():
    rng = np.random.default_rng(42)
    estimates = rng.uniform(0, 1, 200)
    targets = (rng.uniform(0, 1, 200) < estimates).astype(int)

    prob_true, prob_pred = calibration_curve(targets, estimates, num_bins=5, strategy="uniform")
    assert len(prob_true) == len(prob_pred) <= 5
    assert np.all((prob_true >= 0) & (prob_true <= 1))

    prob_true, prob_pred = calibration_curve(targets, estimates, num_bins=5, strategy="quantile")
    assert len(prob_true) == len(prob_pred) <= 5
    # Quantile bins hold roughly equal sample counts, so the mean predicted
    # probabilities must be strictly increasing across bins
    assert np.all(np.diff(prob_pred) > 0)


def test_calibration_curve_invalid_inputs():
    targets = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError, match="outside"):
        calibration_curve(targets, np.array([0.1, 1.2, 0.3, 0.4]))

    with pytest.raises(ValueError, match="binary"):
        calibration_curve(np.array([0, 1, 2, 1]), np.array([0.1, 0.2, 0.3, 0.4]))

    with pytest.raises(ValueError, match="strategy"):
        calibration_curve(targets, np.array([0.1, 0.2, 0.3, 0.4]), strategy="invalid")
