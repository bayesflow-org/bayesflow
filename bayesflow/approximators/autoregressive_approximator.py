from functools import wraps

from bayesflow.estimators import AutoregressiveEstimator
from bayesflow.utils.serialization import serializable

from ._deprecation import warn_approximator_deprecation


@serializable("bayesflow.approximators")
class AutoregressiveApproximator(AutoregressiveEstimator):
    """Deprecated alias for :class:`bayesflow.estimators.AutoregressiveEstimator`."""

    @wraps(AutoregressiveEstimator.__init__)
    def __init__(self, *args, **kwargs):
        warn_approximator_deprecation("AutoregressiveApproximator", "AutoregressiveEstimator")
        super().__init__(*args, **kwargs)
