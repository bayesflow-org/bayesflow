r"""
Post-hoc corrections of amortized posterior draws, e.g., by Pareto-smoothed importance sampling (PSIS).
"""

from .psis import pareto_smooth, psis_correction

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
