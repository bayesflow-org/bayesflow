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

>>> # A network trained with the Brier scoring rule.
>>> from bayesflow.scoring_rules import BrierScore
>>> brier_network = bf.networks.ScoringRuleNetwork(scoring_rule=BrierScore())

>>> # A network trained with the polynomial (Tsallis) scoring rule.
>>> from bayesflow.scoring_rules import PolynomialScore
>>> poly_network = bf.networks.ScoringRuleNetwork(scoring_rule=PolynomialScore(alpha=2.0))

>>> # A network trained with the exponential scoring rule to estimate log Bayes factors.
>>> from bayesflow.scoring_rules import ExponentialScore
>>> exp_network = bf.networks.ScoringRuleNetwork(scoring_rule=ExponentialScore(leaky=2.0))

>>> # A network trained with the logistic scoring rule to estimate log Bayes factors.
>>> from bayesflow.scoring_rules import LogisticScore
>>> logistic_network = bf.networks.ScoringRuleNetwork(scoring_rule=LogisticScore())

Inherit from :py:class:`ScoringRule` to build your own custom scoring rule.
"""

from .scoring_rule import ScoringRule
from .parametric_distribution_score import ParametricDistributionScore
from .normed_difference_score import NormedDifferenceScore
from .mixture_score import MixtureScore
from .mean_score import MeanScore
from .median_score import MedianScore
from .quantile_score import QuantileScore
from .mv_normal_score import MvNormalScore
from .cross_entropy_score import CrossEntropyScore
from .brier_score import BrierScore
from .polynomial_score import PolynomialScore
from .exponential_score import ExponentialScore
from .logistic_score import LogisticScore

from bayesflow.utils._docs import _add_imports_to_all

_add_imports_to_all()
