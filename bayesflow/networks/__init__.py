r"""
A rich collection of neural network architectures for use in :py:class:`~bayesflow.approximators.Approximator`\ s.

Examples
--------
>>> import bayesflow as bf  # doctest: +SKIP
>>> approximator = bf.ContinuousApproximator(  # doctest: +SKIP
...     inference_network=bf.networks.CouplingFlow(),
...     summary_network=bf.networks.DeepSet(),
... )
"""

# Base classes
from .inference import InferenceNetwork
from .summary import SummaryNetwork

# Inference networks
from . import inference
from .inference import ConsistencyModel, StableConsistencyModel
from .inference import CouplingFlow
from .inference import DiffusionModel
from .inference import FlowMatching
from .inference import LatentInferenceNetwork, Encoder, Decoder
from .inference import LatentDiffusionModel
from .inference import PointNetwork, ScoringRuleNetwork

# Summary networks
from . import summary
from .summary import ConvolutionalNetwork
from .summary import DeepSet
from .summary import FusionNetwork
from .summary import TimeSeriesNetwork
from .summary import SetTransformer, TimeSeriesTransformer, FusionTransformer

# Subnets (backbones for inference / summary networks)
from . import subnets
from .subnets import MLP, TimeMLP
from .subnets import UViT, UNet, ResidualUViT

from . import defaults

__all__ = ["inference", "summary", "subnets", "defaults"]
