"""
This module provides the :py:class:`~bayesflow.simulators.Simulator`, which is the base implementation of a
generative mathematical model, or data generating process.
Its primary function is to sample data with the :py:meth:`~bayesflow.simulators.Simulator.sample` method.
The module also contains several other kinds of Simulators, as well as the function
:py:func:`~bayesflow.simulators.make_simulator` to facilitate easy implementation.
"""

from .sequential_simulator import SequentialSimulator
from .hierarchical_simulator import HierarchicalSimulator
from .lambda_simulator import LambdaSimulator
from .make_simulator import make_simulator
from .model_comparison_simulator import ModelComparisonSimulator
from .simulator import Simulator

from .benchmark_simulators import (
    LotkaVolterra,
    SIR,
    TwoMoons,
)

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
