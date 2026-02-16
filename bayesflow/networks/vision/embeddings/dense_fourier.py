import keras

from bayesflow.networks import Sequential
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.embeddings import FourierEmbedding


@serializable("bayesflow.networks")
class DenseFourier(keras.Layer):
    """
    Time embedding block for diffusion-style architectures (U-Net / U-ViT), mapping a scalar timestep `t` to a global
    conditiong vector.

    Pattern:
        .. code-block:: text

            e   = FourierEmbedding(t)          # sin/cos features (+ optional raw t)
            out = e + MLP(e)                   # optional; final proj is zero-init => starts near fourier embedding

    Shapes:
        t:   (B, 1)
        out: (B, D) if include_identity=False else (B, D+1)
    """
    def __init__(
        self,
        emb_dim: int = 32,
        *,
        # fourier embedding
        fourier_scale: float = 30.0,
        fourier_initializer: str = "random_normal",
        fourier_trainable: bool = True,
        include_identity: bool = True,
        # residual MLP
        use_residual_mlp: bool = True,
        activation: str = "mish",
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        **kwargs,
    ):
        """
        Implements a time embedding block used for global conditioning in vision diffusion backbones.

        Parameters
        ----------
        emb_dim : int
            Fourier feature dim `D` (even). Output is `D` (no identity) or `D+1` (with identity).
        fourier_scale : float
            Frequency scale (period range). Default 30.0.
        fourier_initializer : str
            Initializer for Fourier frequencies. Default "random_normal".
        fourier_trainable : bool
            Whether Fourier frequencies are trainable. Default True.
        include_identity : bool
            If True, includes raw `t` as an extra coordinate. Default True.
        use_residual_mlp : bool
            If True, uses `out = e + MLP(e)` with zero-init last layer. Default True.
        activation : str
            Residual MLP activation. Default "mish".
        kernel_initializer : str | keras.Initializer
            Init for residual MLP (except final zero-init). Default "he_normal".
        **kwargs
            Passed to `keras.Layer` (e.g., name, dtype, trainable).
        """
        super().__init__(**layer_kwargs(kwargs))

        self.emb_dim = int(emb_dim)

        self.fourier_scale = float(fourier_scale)
        self.fourier_initializer = str(fourier_initializer)
        self.fourier_trainable = bool(fourier_trainable)
        self.include_identity = bool(include_identity)

        self.use_residual_mlp = bool(use_residual_mlp)
        self.activation = str(activation)
        self.kernel_initializer = kernel_initializer

        self.fourier = FourierEmbedding(
            embed_dim=self.emb_dim,
            scale=self.fourier_scale,
            initializer=self.fourier_initializer,
            trainable=self.fourier_trainable,
            include_identity=self.include_identity,
            name="fourier",
        )

        self.time_mlp = None
        if self.use_residual_mlp:
            self.time_mlp = Sequential([
                keras.layers.Dense(
                    self.emb_dim + (1 if self.include_identity else 0),
                    activation=self.activation,
                    kernel_initializer=self.kernel_initializer,
                    name="time_mlp_1",
                ),
                keras.layers.Dense(
                    self.emb_dim + (1 if self.include_identity else 0),
                    kernel_initializer="zeros",
                    name="time_mlp_2",
                )
            ])

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "emb_dim": self.emb_dim,
            "fourier_scale": self.fourier_scale,
            "fourier_initializer": self.fourier_initializer,
            "fourier_trainable": self.fourier_trainable,
            "include_identity": self.include_identity,
            "use_residual_mlp": self.use_residual_mlp,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        self.fourier.build(input_shape)
        emb_shape = self.fourier.compute_output_shape(input_shape)
        if self.time_mlp is not None:
            self.time_mlp.build(emb_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        emb_shape = self.fourier.compute_output_shape(input_shape)
        return emb_shape

    def call(self, t: Tensor, training: bool | None = None, **kwargs) -> Tensor:
        t_emb = self.fourier(t)
        if self.time_mlp is not None:
            delta = self.time_mlp(t_emb, training=training)
            t_emb = t_emb + delta
        return t_emb
