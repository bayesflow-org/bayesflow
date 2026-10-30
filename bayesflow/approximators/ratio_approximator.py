from functools import wraps

from bayesflow.estimators import RatioEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class RatioApproximator(RatioEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.RatioEstimator`."""

    @wraps(RatioEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("RatioApproximator", "RatioEstimator")
        super().__init__(*args, **kwargs)
