from functools import wraps
from warnings import warn

from bayesflow.experimental.multilevel_estimator import MultilevelEstimator
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.experimental")
class GraphicalApproximator(MultilevelEstimator):
    """Deprecated alias for :class:`~bayesflow.experimental.multilevel_estimator.MultilevelEstimator`."""

    @wraps(MultilevelEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn(
            "GraphicalApproximator is deprecated. Use "
            "bayesflow.experimental.multilevel_estimator.MultilevelEstimator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
