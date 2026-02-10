"""
This module provides :py:class:`~bayesflow.simulators.Simulator`, :py:func:`~bayesflow.simulators.make_simulator`,
and several other kinds of :py:class:`~bayesflow.simulators.Simulator` implementations, which serve as generative
mathematical models, or data generating processes, with their primary function being to sample data.
"""

from .sequential_simulator import SequentialSimulator
from .hierarchical_simulator import HierarchicalSimulator
from .lambda_simulator import LambdaSimulator
from .make_simulator import make_simulator
from .model_comparison_simulator import ModelComparisonSimulator
from .simulator import Simulator

from .benchmark_simulators import (
    BernoulliGLM,
    BernoulliGLMRaw,
    GaussianLinear,
    GaussianLinearUniform,
    GaussianMixture,
    InverseKinematics,
    LotkaVolterra,
    SIR,
    SLCP,
    SLCPDistractors,
    TwoMoons,
)

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
