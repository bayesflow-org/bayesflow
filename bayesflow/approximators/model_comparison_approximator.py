from functools import wraps

from bayesflow.estimators import ModelComparisonEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class ModelComparisonApproximator(ModelComparisonEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.ModelComparisonEstimator`."""

    @wraps(ModelComparisonEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("ModelComparisonApproximator", "ModelComparisonEstimator")
        super().__init__(*args, **kwargs)
