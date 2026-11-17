from functools import wraps

from bayesflow.estimators import BasicEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class ContinuousApproximator(BasicEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.BasicEstimator`."""

    @wraps(BasicEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("ContinuousApproximator", "BasicEstimator")
        super().__init__(*args, **kwargs)
