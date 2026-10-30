"""Deprecated aliases for :mod:`bayesflow.estimators`."""

from .approximator import Approximator
from .continuous_approximator import ContinuousApproximator
from .autoregressive_approximator import AutoregressiveApproximator
from .compositional_approximator import CompositionalApproximator
from .scoring_rule_approximator import ScoringRuleApproximator
from .model_comparison_approximator import ModelComparisonApproximator
from .ratio_approximator import RatioApproximator
from .ensemble_approximator import EnsembleApproximator
from . import helpers as helpers

__all__ = [
    "Approximator",
    "ContinuousApproximator",
    "AutoregressiveApproximator",
    "CompositionalApproximator",
    "ScoringRuleApproximator",
    "ModelComparisonApproximator",
    "RatioApproximator",
    "EnsembleApproximator",
]
