r"""
A collection of :py:class:`~bayesflow.estimators.Estimator`\ s, which embody the inference task and the
neural network components used to perform it.
"""

from .estimator import Estimator
from .basic_estimator import BasicEstimator
from .autoregressive_estimator import AutoregressiveEstimator
from .compositional_estimator import CompositionalEstimator
from .scoring_rule_estimator import ScoringRuleEstimator
from .model_comparison_estimator import ModelComparisonEstimator
from .ratio_estimator import RatioEstimator
from .ensemble_estimator import EnsembleEstimator
from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
