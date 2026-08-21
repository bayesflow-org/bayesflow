from bayesflow.types import Tensor
from ..invertible_layer import InvertibleLayer


class Transform(InvertibleLayer):
    """Base class for the elementwise bijections applied inside a coupling layer.

    A transform declares how many parameters it needs per target dimension, splits
    the flat parameter vector predicted by the coupling subnet into named parts,
    constrains those parts to their valid ranges, and applies the resulting
    bijection.

    Subclasses implement ``params_per_dim``, ``split_parameters``,
    ``constrain_parameters``, ``_forward`` and ``_inverse``.
    """

    @property
    def params_per_dim(self) -> int:
        raise NotImplementedError

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def call(self, xz: Tensor, parameters: dict[str, Tensor], inverse: bool = False) -> (Tensor, Tensor):
        if inverse:
            return self._inverse(xz, parameters)
        return self._forward(xz, parameters)

    def _forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError
