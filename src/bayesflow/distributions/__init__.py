r"""
A collection of :py:class:`~bayesflow.distributions.Distribution`\ s,
which represent the latent space for :py:class:`~bayesflow.networks.InferenceNetwork`\ s
or the summary space of :py:class:`~bayesflow.networks.SummaryNetwork`\ s.
"""

from .distribution import Distribution
from .diagonal_normal import DiagonalNormal
from .diagonal_student_t import DiagonalStudentT
from .mixture import Mixture

from ..utils._docs import _add_imports_to_all

_add_imports_to_all(include_modules=[])
