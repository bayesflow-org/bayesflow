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
from .parametric_distribution_scoring_rule import ParametricDistributionScore
from .normed_difference_scoring_rule import NormedDifferenceScore
from .mean_scoring_rule import MeanScore
from .median_scoring_rule import MedianScore
from .quantile_scoring_rule import QuantileScore
from .mv_normal_scoring_rule import MvNormalScore

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
