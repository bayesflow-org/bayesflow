r"""
Generative neural networks for approximating conditional distributions.
"""

from .inference_network import InferenceNetwork
from .consistency import ConsistencyModel, StableConsistencyModel
from .coupling import CouplingFlow
from .diffusion import DiffusionModel
from .flow_matching import FlowMatching
from .scoring import ScoringRuleNetwork, PointNetwork
from .scoring import (
    ScoringRule,
    MeanScore,
    MedianScore,
    QuantileScore,
    MvNormalScore,
    NormedDifferenceScore,
    CrossEntropyScore,
)
