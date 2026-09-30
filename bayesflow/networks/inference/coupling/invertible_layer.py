import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs

from bayesflow.utils.serialization import deserialize


class InvertibleLayer(keras.Layer):
    """Base class for the layers a :py:class:`~bayesflow.networks.CouplingFlow` is composed of.

    Subclasses implement ``call`` with an ``inverse`` flag selecting between
    ``_forward`` and ``_inverse``, and return the transformed tensor together with
    the log determinant of the Jacobian.
    """

    def __init__(self, **kwargs):
        super().__init__(**layer_kwargs(kwargs))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def call(self, *args, **kwargs):
        # we cannot provide a default implementation for this
        #  because the signature of layer.call() is used to
        #  determine the arguments to layer.build()
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _inverse(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _skip_fixed_dims(
        inputs: Tensor, outputs: Tensor, log_jac: Tensor, fixed_target_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        """Return outputs and the log-det and honor the ``fixed_target_mask``:
        Fixed dims are unchanged and don't affect the log-det."""
        if fixed_target_mask is not None:
            outputs = keras.ops.where(fixed_target_mask, outputs, inputs)
            log_jac = keras.ops.where(fixed_target_mask, log_jac, 0.0)
        return outputs, keras.ops.sum(log_jac, axis=-1)
