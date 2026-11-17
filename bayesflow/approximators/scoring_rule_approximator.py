from functools import wraps

from bayesflow.estimators import ScoringRuleEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class ScoringRuleApproximator(ScoringRuleEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.ScoringRuleEstimator`."""

    @wraps(ScoringRuleEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("ScoringRuleApproximator", "ScoringRuleEstimator")
        super().__init__(*args, **kwargs)
