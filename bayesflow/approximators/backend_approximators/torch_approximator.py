from functools import wraps

from bayesflow.estimators.backend_estimators.torch_estimator import TorchEstimator

from .._deprecation import warn_approximator_deprecation


class TorchApproximator(TorchEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.backend_estimators.TorchEstimator`."""

    @wraps(TorchEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("backend_approximators.TorchApproximator", "backend_estimators.TorchEstimator")
        super().__init__(*args, **kwargs)
