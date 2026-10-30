from functools import wraps

from bayesflow.estimators.backend_estimators.tensorflow_estimator import TensorFlowEstimator

from .._deprecation import warn_approximator_deprecation


class TensorFlowApproximator(TensorFlowEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.backend_estimators.TensorFlowEstimator`."""

    @wraps(TensorFlowEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation(
            "backend_approximators.TensorFlowApproximator", "backend_estimators.TensorFlowEstimator"
        )
        super().__init__(*args, **kwargs)
