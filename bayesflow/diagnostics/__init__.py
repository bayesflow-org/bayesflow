r"""
A collection of plotting utilities and metrics for evaluating trained :py:class:`~bayesflow.workflows.Workflow`\ s.
"""

from .metrics import (
    bootstrap_comparison,
    calibration_error,
    posterior_contraction,
    summary_space_comparison,
)

from .plots import (
    calibration_ecdf,
    calibration_ecdf_from_quantiles,
    calibration_histogram,
    loss,
    mc_calibration,
    mc_confusion_matrix,
    mmd_hypothesis_test,
    pairs_posterior,
    pairs_samples,
    recovery,
    recovery_from_estimates,
    z_score_contraction,
)

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
