"""
Custom types used for type annotations.

.. currentmodule:: bayesflow.types

.. autodata:: LogProb
.. autodata:: Shape
.. autodata:: Tensor
"""

from .log_prob import LogProb
from .shape import Shape
from .tensor import Tensor

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
