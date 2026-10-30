from functools import wraps

from bayesflow.estimators import Estimator

from ._deprecation import warn_approximator_deprecation


class Approximator(Estimator):
    """Deprecated alias for :class:`bayesflow.estimators.Estimator`."""

    @wraps(Estimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("Approximator", "Estimator")
        super().__init__(*args, **kwargs)
