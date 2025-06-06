r"""
A collection of `keras.Metric <https://keras.io/api/metrics/base_metric/#metric-class>`__\ s for evaluating the
performance of models.
"""

from . import functional
from .maximum_mean_discrepancy import MaximumMeanDiscrepancy
from .root_mean_squard_error import RootMeanSquaredError

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=["functional"])
