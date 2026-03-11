r"""
A rich collection of neural network architectures for use in :py:class:`~bayesflow.approximators.Approximator`\ s.

The module features inference networks (IN), summary networks (SN), as well as general purpose components.
"""

# Base classes
from .inference_network import InferenceNetwork
from .summary_network import SummaryNetwork

# Inference networks
from .inference import ConsistencyModel, StableConsistencyModel
from .inference import CouplingFlow
from .inference import DiffusionModel
from .inference import FlowMatching
from .inference import PointNetwork, ScoringRuleNetwork

# Summary networks
from .summary import ConvolutionalNetwork
from .summary import DeepSet
from .summary import FusionNetwork
from .summary import TimeSeriesNetwork
from .summary import SetTransformer, TimeSeriesTransformer, FusionTransformer

# Subnets (backbones for inference / summary networks)
from .subnets import MLP, TimeMLP
from .subnets import UViT, UNet, ResidualUViT

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=["inference.diffusion"])
