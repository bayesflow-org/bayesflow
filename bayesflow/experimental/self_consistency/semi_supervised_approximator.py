from functools import wraps
from warnings import warn

from bayesflow.utils.serialization import serializable

from .semi_supervised_estimator import SemiSupervisedEstimator


@serializable("bayesflow.experimental")
class SemiSupervisedApproximator(SemiSupervisedEstimator):
    """Deprecated alias for :class:`SemiSupervisedEstimator`."""

    @wraps(SemiSupervisedEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn(
            "SemiSupervisedApproximator is deprecated. Use "
            "bayesflow.experimental.self_consistency.SemiSupervisedEstimator instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if "approximators" in kwargs:
            if "estimators" in kwargs:
                raise TypeError("Received both deprecated 'approximators' and replacement 'estimators'.")
            kwargs["estimators"] = kwargs.pop("approximators")
        super().__init__(*args, **kwargs)

    @wraps(SemiSupervisedEstimator.compile)
    def compile(self, *args, **kwargs):
        if "approximator_metrics" in kwargs:
            if "estimator_metrics" in kwargs:
                raise TypeError("Received both deprecated 'approximator_metrics' and replacement 'estimator_metrics'.")
            kwargs["estimator_metrics"] = kwargs.pop("approximator_metrics")
        return super().compile(*args, **kwargs)

    def sample(self, *, approximator: str | None = None, estimator: str | None = None, **kwargs):
        if approximator is not None:
            if estimator is not None:
                raise TypeError("Received both deprecated 'approximator' and replacement 'estimator'.")
            estimator = approximator
        if estimator is None:
            raise TypeError("Missing required keyword argument: 'approximator'.")
        return super().sample(estimator=estimator, **kwargs)

    @property
    def approximators(self):
        return self.estimators

    @approximators.setter
    def approximators(self, value):
        self.estimators = value

    @property
    def approximator_metrics(self):
        return self.estimator_metrics

    @approximator_metrics.setter
    def approximator_metrics(self, value):
        self.estimator_metrics = value
