import keras
from typing import Literal, Callable, Sequence

from bayesflow.utils import concatenate_valid
from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.types import Tensor


@serializable("bayesflow.networks")
class RandomFourierTimeEmbedding(keras.Layer):
    """Random Fourier features time embedding."""

    def __init__(self, embed_dim: int, fourier_scale: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.fourier_scale = fourier_scale
        self.fourier_weights = self.add_weight(
            name="fourier_weights",
            shape=(1, self.embed_dim // 2),
            initializer=keras.initializers.RandomNormal(stddev=self.fourier_scale),
            trainable=False,
        )

    def compute_output_shape(self, input_shape):
        """Compute output shape.

        Parameters
        ----------
        input_shape : tuple
            Shape of input tensor.

        Returns
        -------
        tuple
            Output shape (batch_size, embed_dim).
        """
        return input_shape[0], self.embed_dim

    def call(self, t):
        # t shape: (batch_size, 1) or (batch_size,)
        if len(t.shape) == 1:
            t = keras.ops.expand_dims(t, -1)

        args = 2 * 3.14159265359 * t * self.fourier_weights
        embedding = keras.ops.concatenate([keras.ops.sin(args), keras.ops.cos(args)], axis=-1)

        # Handle odd embed_dim
        if self.embed_dim % 2 == 1:
            embedding = keras.ops.pad(embedding, [[0, 0], [0, 1]])

        return embedding

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim, "fourier_scale": self.fourier_scale})
        return config


@serializable("bayesflow.networks")
class TimeMLP(keras.Layer):
    """
    Implements a time-conditioned multi-layer perceptron (MLP).

    The model processes three separate inputs: the state variable `x`, a scalar or vector-valued time input `t`,
    and a conditioning variable `conditions`. The input and condition are projected into a shared feature space,
    merged, and passed through a deep residual MLP. A learned time embedding is injected additively at every
    hidden layer.

    If `residual` is enabled, each layer includes a skip connection for improved gradient flow. The model also
        supports dropout for regularization and spectral normalization for stability in learning smooth functions.
    """

    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        widths: Sequence[int] = (256, 256),
        time_emb_dim: int = 32,
        *,
        time_emb: keras.layers.Layer | None = None,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.layers.Layer | None = "layer",
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        Implements a time-conditioned multi-layer perceptron (MLP).

        Parameters
        ----------
        input_dim : int
            Dimensionality of the input variable `x`.
        condition_dim : int
            Dimensionality of the conditioning variable.
        time_emb_dim : int, optional
            Dimensionality of the learned time embedding. Default is 32.
        widths : Sequence[int], optional
            Defines the number of hidden units per layer, as well as the number of layers to be used.
        time_emb : keras.layers.Layer or None, optional
            Custom time embedding layer. If None, a random Fourier feature embedding is used.
        activation : str, optional
            Activation function applied in the hidden layers, such as "mish". Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        residual : bool, optional
            Whether to use residual connections for improved training stability. Default is True.
        dropout : float or None, optional
            Dropout rate applied within the MLP layers for regularization. Default is 0.05.
        norm : str or keras.layers.Layer or None, optional
            Normalization applied after each hidden layer ("batch", "layer", or None). Default is "layer".
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to dense layers. Default is False.
        **kwargs
            Additional keyword arguments passed to `keras.Model`.
        """
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        self.widths = list(widths)
        self.num_layers = len(self.widths)

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization

        # Input projections
        self.input_layer = keras.layers.Dense(
            self.widths[0],
            kernel_initializer=self.kernel_initializer,
        )
        self.condition_layer = keras.layers.Dense(
            self.widths[0],
            kernel_initializer=self.kernel_initializer,
        )
        self.input_merge_layer = keras.layers.Dense(
            self.widths[0],
            kernel_initializer=self.kernel_initializer,
        )

        # Time embedding
        if time_emb is None:
            self.time_emb = RandomFourierTimeEmbedding(
                embed_dim=self.time_emb_dim,
                fourier_scale=kwargs.get("fourier_scale", 30.0),
            )
        else:
            self.time_emb = time_emb

        self.act = keras.layers.Activation(keras.activations.get(self.activation))

        # Core time-conditioned blocks
        self._dense_layers = []
        self._time_proj_layers = []
        self._dropout_layers = []
        self._activation_layers = []
        self._norm_layers = []

        for width in widths:
            dense, time_proj, dropout_layer, activation_layer, norm_layer = self._make_block(
                width=width,
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                dropout=self.dropout,
                norm=self.norm,
                spectral_normalization=self.spectral_normalization,
            )

            self._dense_layers.append(dense)
            self._time_proj_layers.append(time_proj)
            self._dropout_layers.append(dropout_layer)
            self._activation_layers.append(activation_layer)
            self._norm_layers.append(norm_layer)

        # Output projection
        self.output_layer = keras.layers.Dense(
            self.input_dim,
            kernel_initializer=keras.initializers.Zeros(),
        )

    @staticmethod
    def _make_block(
        width: int,
        *,
        activation,
        kernel_initializer,
        dropout,
        norm,
        spectral_normalization: bool,
    ):
        """
        Constructs a single time-conditioned MLP block.

        Returns
        -------
        tuple
            A tuple containing:
            (dense_layer, time_projection_layer, dropout_layer, activation_layer, normalization_layer)
        """

        dense = keras.layers.Dense(width, kernel_initializer=kernel_initializer)
        if spectral_normalization:
            dense = keras.layers.SpectralNormalization(dense)

        time_proj = keras.layers.Dense(width, kernel_initializer=kernel_initializer)

        if dropout is not None and dropout > 0.0:
            dropout_layer = keras.layers.Dropout(dropout)
        else:
            dropout_layer = None

        activation = keras.activations.get(activation)
        if not isinstance(activation, keras.Layer):
            activation = keras.layers.Activation(activation)

        if norm == "batch":
            norm_layer = keras.layers.BatchNormalization()
        elif norm == "layer":
            norm_layer = keras.layers.LayerNormalization()
        elif isinstance(norm, keras.layers.Layer):
            norm_layer = norm
        else:
            norm_layer = None

        return dense, time_proj, dropout_layer, activation, norm_layer

    def build(self, x_shape=None, t_shape=None, conditions_shape=None):
        """
        Builds the model layers given input shapes.

        Parameters
        ----------
        x_shape : tuple, optional
            Shape of the input variable `x`.
        t_shape : tuple, optional
            Shape of the time input `t`.
        conditions_shape : tuple, optional
            Shape of the conditioning variable.
        """
        if self.built:
            return

        if isinstance(x_shape, (tuple, list)) and t_shape is None and conditions_shape is None:
            if len(x_shape) == 2:
                x_shape = tuple(x_shape)
                t_shape = (x_shape[0], 1)
                conditions_shape = (x_shape[0], self.condition_dim)
            elif len(x_shape) == 3 and all(isinstance(s, (tuple, list)) for s in x_shape):
                x_shape, t_shape, conditions_shape = x_shape  # type: ignore[misc]

        if x_shape is None:
            x_shape = (None, self.input_dim)
        if t_shape is None:
            t_shape = (x_shape[0], 1)
        if conditions_shape is None:
            conditions_shape = (x_shape[0], self.condition_dim)

        self.input_layer.build(x_shape)
        self.condition_layer.build(conditions_shape)

        x_emb_shape = self.input_layer.compute_output_shape(x_shape)
        c_emb_shape = self.condition_layer.compute_output_shape(conditions_shape)

        merged_shape = (x_shape[0], int(x_emb_shape[-1]) + int(c_emb_shape[-1]))
        self.input_merge_layer.build(merged_shape)
        h_shape = self.input_merge_layer.compute_output_shape(merged_shape)

        self.time_emb.build(t_shape)
        t_emb_shape = self.time_emb.compute_output_shape(t_shape)

        core_h_shape = h_shape
        for i in range(self.num_layers):
            self._dense_layers[i].build(core_h_shape)
            core_h_shape = self._dense_layers[i].compute_output_shape(core_h_shape)

            self._time_proj_layers[i].build(t_emb_shape)

            if self._dropout_layers[i] is not None:
                self._dropout_layers[i].build(core_h_shape)

            if self._norm_layers[i] is not None:
                self._norm_layers[i].build(core_h_shape)

        self.output_layer.build(core_h_shape)

        super().build(x_shape)

    def compute_output_shape(self, x_shape=None, t_shape=None, conditions_shape=None):
        """
        Computes the output shape of the model.
        """
        if x_shape is None:
            return None, self.input_dim
        return x_shape[0], self.input_dim

    def call(self, x: Tensor, t: Tensor, conditions: Tensor, training=None):
        """
        Evaluates the time-conditioned MLP.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape `(batch_size, input_dim)`.
        t : Tensor
            Time input of shape `(batch_size,)` or `(batch_size, 1)`.
        conditions : Tensor
            Conditioning variable of shape `(batch_size, condition_dim)`.
        training : bool, optional
            Whether the model is in training mode.

        Returns
        -------
        Tensor
            Output tensor of shape `(batch_size, input_dim)`.
        """
        x_emb = self.input_layer(x)
        c_emb = self.condition_layer(conditions)

        merged = concatenate_valid([x_emb, c_emb], axis=-1)
        h = self.input_merge_layer(self.act(merged))
        h = self.act(h)

        t_emb = self.time_emb(t)

        for i in range(self.num_layers):
            h_old = h

            h = self._dense_layers[i](h)

            if self._dropout_layers[i] is not None:
                h = self._dropout_layers[i](h, training=training)

            h = self._activation_layers[i](h)
            h = h + self._time_proj_layers[i](t_emb)

            if self.residual and h_old.shape[-1] == h.shape[-1]:
                h = h + h_old

            if self._norm_layers[i] is not None:
                h = self._norm_layers[i](h, training=training)

        return self.output_layer(h)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # reconstruct nested layers (time_emb, norm) properly
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        """
        Returns the configuration of the model for serialization.
        """
        cfg = super().get_config()
        cfg.update(
            {
                "input_dim": self.input_dim,
                "condition_dim": self.condition_dim,
                "time_emb_dim": self.time_emb_dim,
                "widths": self.widths,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "residual": self.residual,
                "dropout": self.dropout,
                "norm": self.norm,
                "spectral_normalization": self.spectral_normalization,
                "time_emb": self.time_emb,
            }
        )
        return cfg
