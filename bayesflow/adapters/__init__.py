"""
A collection of :py:class:`~bayesflow.adapters.Adapter` transforms, which tell BayesFlow how to interpret your
:py:class:`~bayesflow.simulators.Simulator` output and plug it into neural networks for training and inference.
"""

from . import transforms
from .adapter import Adapter

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=["transforms"])
