r"""
A collection of scoring rules for Bayes risk minimization with
:py:class:`~bayesflow.networks.ScoringRuleInferenceNetwork`.

Examples
--------
>>> # A network to estimate both point estimates and parameters of a multivariate normal distribution.
>>> inference_network = bf.networks.ScoringRuleInferenceNetwork(
        scores=dict(
            mean=bf.scores.MeanScore(),
            quantiles=bf.scores.QuantileScore(),
            mvn=bf.scores.MultivariateNormalScore(),
        )
    )

Inherit from :py:class:`ScoringRule` to build your own custom scoring rule.
"""

from .scoring_rule import ScoringRule
from .parametric_distribution_score import ParametricDistributionScore
from .normed_difference_score import NormedDifferenceScore
from .mean_score import MeanScore
from .median_score import MedianScore
from .quantile_score import QuantileScore
from .multivariate_normal_score import MultivariateNormalScore

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
