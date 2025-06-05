r"""
A collection of :py:class:`~bayesflow.approximators.Approximator`\ s, which embody the inference task and the
neural network components used to perform it.
"""

from .approximator import Approximator
from .continuous_approximator import ContinuousApproximator
from .point_approximator import PointApproximator
from .model_comparison_approximator import ModelComparisonApproximator

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
