import keras

from bayesflow.types import Tensor
from ..invertible_layer import InvertibleLayer
from ..masks import skip_fixed_dims


class Transform(InvertibleLayer):
    """Base class for the elementwise bijections applied inside a coupling layer.

    A transform declares how many parameters it needs per target dimension, splits
    the flat parameter vector predicted by the coupling subnet into named parts,
    constrains those parts to their valid ranges, and applies the resulting
    bijection.

    Subclasses implement ``params_per_dim``, ``split_parameters``,
    ``constrain_parameters``, ``_forward`` and ``_inverse``. The latter two return
    the transformed values and their elementwise log-Jacobian;
    ``call`` reduces it to the log-determinant.
    """

    @property
    def params_per_dim(self) -> int:
        raise NotImplementedError

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        raise NotImplementedError

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        raise NotImplementedError

    def call(
        self,
        xz: Tensor,
        parameters: dict[str, Tensor],
        inverse: bool = False,
        fixed_target_mask: Tensor = None,
    ) -> (Tensor, Tensor):
        if inverse:
            out, log_jac = self._inverse(xz, parameters)
        else:
            out, log_jac = self._forward(xz, parameters)
        out, log_jac = skip_fixed_dims(xz, out, log_jac, fixed_target_mask)
        return out, keras.ops.sum(log_jac, axis=-1)

    def _forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        raise NotImplementedError
