r"""
A rich collection of neural network architectures for use in :py:class:`~bayesflow.approximators.Approximator`\ s.

The module features inference networks (IN), summary networks (SN), as well as general purpose networks.
"""

from .consistency_models import ConsistencyModel
from .coupling_flow import CouplingFlow
from .deep_set import DeepSet
from .diffusion_model import DiffusionModel, CompositionalDiffusionModel
from .flow_matching import FlowMatching
from .inference_network import InferenceNetwork
from .point_inference_network import PointInferenceNetwork
from .mlp import MLP, TimeMLP
from .fusion_network import FusionNetwork
from .sequential import Sequential
from .summary_network import SummaryNetwork
from .time_series_network import TimeSeriesNetwork
from .transformers import SetTransformer, TimeSeriesTransformer, FusionTransformer

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=["diffusion_model"])
