# Backward-compatibility shim: scoring_rules now lives at
# bayesflow.networks.inference.scoring.scoring_rules
from bayesflow.networks.inference.scoring.scoring_rules import (  # noqa: F401
    ScoringRule,
    CategoricalScoringRule,
    ParametricDistributionScore,
    NormedDifferenceScore,
    MeanScore,
    MedianScore,
    QuantileScore,
    MvNormalScore,
    MixtureScore,
    CrossEntropyScore,
    BrierScore,
    PolynomialScore,
    ExponentialScore,
    LogisticScore,
    __doc__,
)

from bayesflow.utils._docs import _add_imports_to_all

_add_imports_to_all()
