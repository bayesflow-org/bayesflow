from bayesflow.utils.serialization import serializable
from bayesflow.types import Tensor

from ..summary_network import SummaryNetwork


@serializable("bayesflow.networks")
class Transformer(SummaryNetwork):
    """Abstract base class for transformer models that can server as summary networks."""

    def call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        raise NotImplementedError
