from functools import wraps

from bayesflow.estimators.backend_estimators import BackendEstimator

from .._deprecation import warn_approximator_deprecation


class BackendApproximator(BackendEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.backend_estimators.BackendEstimator`."""

    @wraps(BackendEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation(
            "backend_approximators.BackendApproximator", "backend_estimators.BackendEstimator"
        )
        super().__init__(*args, **kwargs)
