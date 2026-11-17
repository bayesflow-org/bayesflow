from bayesflow.estimators import EnsembleEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class EnsembleApproximator(EnsembleEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.EnsembleEstimator`."""

    def __init__(self, approximators=None, *, estimators=None, **kwargs):
        warn_approximator_deprecation("EnsembleApproximator", "EnsembleEstimator")
        if approximators is not None and estimators is not None:
            raise TypeError("Received both deprecated 'approximators' and replacement 'estimators'.")
        estimators = approximators if approximators is not None else estimators
        if estimators is None:
            raise TypeError("Missing required argument: 'approximators'.")
        super().__init__(estimators=estimators, **kwargs)

    @property
    def approximators(self):
        return self.estimators

    @approximators.setter
    def approximators(self, value):
        self.estimators = value
