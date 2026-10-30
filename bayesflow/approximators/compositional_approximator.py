from functools import wraps

from bayesflow.estimators import CompositionalEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class CompositionalApproximator(CompositionalEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.CompositionalEstimator`."""

    @wraps(CompositionalEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("CompositionalApproximator", "CompositionalEstimator")
        super().__init__(*args, **kwargs)
