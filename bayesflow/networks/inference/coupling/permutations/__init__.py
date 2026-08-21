r"""
Permutations applied between the coupling layers of a
:py:class:`~bayesflow.networks.CouplingFlow`.
"""

from .orthogonal import OrthogonalPermutation
from .fixed_permutation import FixedPermutation
from .random import RandomPermutation
from .swap import Swap

from bayesflow.utils._docs import _add_imports_to_all

_add_imports_to_all()
