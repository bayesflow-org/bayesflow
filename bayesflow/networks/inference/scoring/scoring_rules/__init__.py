r"""
A collection of scoring rules for Bayes risk minimization with
:py:class:`~bayesflow.networks.ScoringRuleNetwork`.

Examples
--------
>>> # A network to estimate both point estimates and parameters of a multivariate normal distribution.
>>> from bayesflow.scoring_rules import MeanScore, QuantileScore, MvNormalScore
>>> import bayesflow as bf
>>> inference_network = bf.networks.ScoringRuleNetwork(
...     mean=MeanScore(),
...     quantiles=QuantileScore(),
...     mvn=MvNormalScore(),
... )

>>> # A network to estimate posterior model probabilities with multiple categorical scoring rules.
>>> from bayesflow.scoring_rules import (
...     CrossEntropyScore,
...     BrierScore,
...     PolynomialScore,
...     ExponentialScore,
...     LogisticScore,
... )
>>> comparison_network = bf.networks.ScoringRuleNetwork(
...     cross_entropy=CrossEntropyScore(),
...     brier=BrierScore(),
...     polynomial=PolynomialScore(alpha=2.0),
...     exponential=ExponentialScore(links={"logits": bf.links.Leaky(power=2.0)}),
...     logistic=LogisticScore(),
...     power_logistic=LogisticScore(alpha=1.0),
... )

Inherit from :py:class:`ScoringRule` to build your own custom scoring rule.
"""

from .scoring_rule import ScoringRule
from .categorical_scoring_rule import CategoricalScoringRule
from .parametric_distribution_score import ParametricDistributionScore
from .normed_difference_score import NormedDifferenceScore
from .mixture_score import MixtureScore
from .mean_score import MeanScore
from .median_score import MedianScore
from .quantile_score import QuantileScore
from .mv_normal_score import MvNormalScore
from .cross_entropy_score import CrossEntropyScore
from .polynomial_score import PolynomialScore
from .brier_score import BrierScore
from .exponential_score import ExponentialScore
from .logistic_score import LogisticScore

from bayesflow.utils._docs import _add_imports_to_all

_add_imports_to_all()
