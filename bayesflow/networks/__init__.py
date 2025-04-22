"""
A rich collection of neural network architectures for use in :py:class:`~bayesflow.approximators.Approximator`\ s.
"""

from .consistency_models import ConsistencyModel
from .coupling_flow import CouplingFlow
from .deep_set import DeepSet
from .flow_matching import FlowMatching
from .inference_network import InferenceNetwork
from .point_inference_network import PointInferenceNetwork
from .mlp import MLP
from .summary_network import SummaryNetwork
from .time_series_network import TimeSeriesNetwork
from .transformers import SetTransformer, TimeSeriesTransformer, FusionTransformer

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
