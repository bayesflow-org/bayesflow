from functools import wraps

from bayesflow.estimators.backend_estimators.jax_estimator import JAXEstimator

from .._deprecation import warn_approximator_deprecation


class JAXApproximator(JAXEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.backend_estimators.JAXEstimator`."""

    @wraps(JAXEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("backend_approximators.JAXApproximator", "backend_estimators.JAXEstimator")
        super().__init__(*args, **kwargs)
