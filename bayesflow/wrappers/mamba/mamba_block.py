import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.wrappers")
class MambaBlock(keras.Layer):
    """
    Wraps the original Mamba module from, with added functionality for bidirectional processing:
    https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

    Copyright (c) 2023, Tri Dao, Albert Gu.
    """

    def __init__(
        self,
        state_dim: int,
        conv_dim: int,
        feature_dim: int = 16,
        expand: int = 2,
        bidirectional: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        device: str = "cuda",
        **kwargs,
    ):
        """
        A Keras layer implementing a Mamba-based sequence processing block.

        This layer applies a Mamba model for sequence modeling, preceded by a
        convolutional projection and followed by layer normalization.

        Parameters
        ----------
        state_dim : int
            The dimension of the state space in the Mamba model.
        conv_dim : int
            The dimension of the convolutional layer used in Mamba.
        feature_dim : int, optional
            The feature dimension for input projection and Mamba processing (default is 16).
        expand : int, optional
            Expansion factor for Mamba's internal dimension (default is 1).
        dt_min : float, optional
            Minimum delta time for Mamba (default is 0.001).
        dt_max : float, optional
            Maximum delta time for Mamba (default is 0.1).
        device : str, optional
            The device to which the Mamba model is moved, typically "cuda" or "cpu" (default is "cuda").
        **kwargs :
            Additional keyword arguments passed to the `keras.layers.Layer` initializer.
        """

        super().__init__(**layer_kwargs(kwargs))

        if keras.backend.backend() != "torch":
            raise RuntimeError("Mamba is only available using torch backend.")

        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError("Could not import Mamba. Please install it via `pip install mamba-ssm`") from e

        self.bidirectional = bidirectional

        self.mamba = Mamba(
            d_model=feature_dim, d_state=state_dim, d_conv=conv_dim, expand=expand, dt_min=dt_min, dt_max=dt_max
        ).to(device)

        self.input_projector = keras.layers.Conv1D(
            feature_dim,
            kernel_size=1,
            strides=1,
        )
        self.layer_norm = keras.layers.LayerNormalization()

    def call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        """
        Applies the Mamba layer to the input tensor `x`, optionally in a bidirectional manner.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape `(batch_size, sequence_length, input_dim)`.
        training : bool, optional
            Whether the layer should behave in training mode (e.g., applying dropout). Default is False.
        **kwargs : dict
            Additional keyword arguments passed to the internal `_call` method.

        Returns
        -------
        Tensor
            Output tensor of shape `(batch_size, sequence_length, feature_dim)` if unidirectional,
            or `(batch_size, sequence_length, 2 * feature_dim)` if bidirectional.
        """

        out_forward = self._call(x, training=training, **kwargs)
        if self.bidirectional:
            out_backward = self._call(keras.ops.flip(x, axis=-2), training=training, **kwargs)
            return keras.ops.concatenate((out_forward, out_backward), axis=-1)
        return out_forward

    def _call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        x = self.input_projector(x)
        h = self.mamba(x)
        out = self.layer_norm(h + x, training=training, **kwargs)
        return out

    @sanitize_input_shape
    def build(self, input_shape):
        super().build(input_shape)
        self.call(keras.ops.zeros(input_shape))
