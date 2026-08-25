r"""
Neural networks for autoregressive approximators.
"""

from .recurrent_decoder import RecurrentDecoder
from .transformer_decoder import TransformerDecoder

from bayesflow.utils._docs import _add_imports_to_all

_add_imports_to_all()
