r"""
A collection of scoring rules for Bayes risk minimization with
:py:class:`~bayesflow.networks.ScoringRuleNetwork`.

Examples
--------
>>> # A network to estimate both point estimates and parameters of a multivariate normal distribution.
>>> from bayesflow.scoring_rules import MeanScore, QuantileScore, MvNormalScore
>>> inference_network = bf.networks.ScoringRuleNetwork(
...     mean=MeanScore(),
...     quantiles=QuantileScore(),
...     mvn=MvNormalScore(),
... )

Inherit from :py:class:`ScoringRule` to build your own custom scoring rule.
"""

from .scoring_rule import ScoringRule
from .parametric_distribution_score import ParametricDistributionScore
from .normed_difference_score import NormedDifferenceScore
from .mean_score import MeanScore
from .median_score import MedianScore
from .quantile_score import QuantileScore
from .mv_normal_score import MvNormalScore
from .cross_entropy_score import CrossEntropyScore

from bayesflow.utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
