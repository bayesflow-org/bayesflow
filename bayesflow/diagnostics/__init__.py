r"""
A collection of plotting utilities and metrics for evaluating trained :py:class:`~bayesflow.workflows.Workflow`\ s.
"""

from . import metrics
from . import plots

from .metrics import (
    bootstrap_comparison,
    calibration_error,
    calibration_log_gamma,
    classifier_two_sample_test,
    expected_calibration_error,
    gamma_discrepancy,
    gamma_null_distribution,
    accuracy,
    brier_score,
    posterior_contraction,
    posterior_z_score,
    root_mean_squared_error,
    summary_space_comparison,
    correlation,
    accuracy_random_points,
)

from .plots import (
    calibration_ecdf,
    calibration_ecdf_from_quantiles,
    calibration_histogram,
    coverage,
    loss,
    mc_calibration,
    mc_confusion_matrix,
    mmd_hypothesis_test,
    pairs_posterior,
    pairs_quantity,
    pairs_samples,
    plot_quantity,
    recovery,
    recovery_from_estimates,
    z_score_contraction,
)

__all__ = ["metrics", "plots"]
