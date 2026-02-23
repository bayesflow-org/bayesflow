r"""
A collection of scoring rules for Bayes risk minimization with
:py:class:`~bayesflow.networks.ScoringRuleNetwork`.

Examples
--------
>>> # A network to estimate both point estimates and parameters of a multivariate normal distribution.
>>> from bayesflow.scoring_rules import MeanScoringRule, QuantileScoringRule, MvNormalScoringRule
>>> inference_network = bf.networks.ScoringRuleNetwork(
        scoring_rules=dict(
            mean=MeanScoringRule(),
            quantiles=QuantileScoringRule(),
            mvn=MvNormalScoringRule(),
        )
    )

Inherit from :py:class:`ScoringRule` to build your own custom scoring rule.
"""

from .scoring_rule import ScoringRule
from .parametric_distribution_scoring_rule import ParametricDistributionScoringRule
from .normed_difference_scoring_rule import NormedDifferenceScoringRule
from .mean_scoring_rule import MeanScoringRule
from .median_scoring_rule import MedianScoringRule
from .quantile_scoring_rule import QuantileScoringRule
from .mv_normal_scoring_rule import MvNormalScoringRule

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
