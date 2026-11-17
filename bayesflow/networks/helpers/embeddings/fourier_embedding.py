import numpy as np

import keras
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class FourierEmbedding(keras.Layer):
    """Implements a Fourier projection with normally distributed frequencies [1].

    Each of the ``D`` time channels is embedded with its own frequencies.

    [1] Tancik et al. (2020), Fourier features let networks learn high frequency functions in low dimensional domains
    """

    def __init__(
        self,
        embed_dim: int = 32,
        scale: float = 30.0,
        initializer: str = "random_normal",
        trainable: bool = True,
        include_identity: bool = True,
        **kwargs,
    ):
        """
        Initialize a Fourier projection instance with normally distributed frequencies.

        Parameter
        ---------
        embed_dim        : int (even)
            Dimensionality of the Fourier projection per time channel. The resulting embedding
            has dimensionality `D * (embed_dim + 1)` if `include_identity` is set to True.
        scale            : float, optional (default - 30.0)
            Scaling factor for the frequencies.
        initializer      : str, optional (default - "random_normal")
            Method for initializing the projection weights.
        trainable        : bool, optional (default - True)
            If True, the projection weights are trainable.
        include_identity : bool, optional (default - True)
            If True, adds an identity mapping component to the embedding.
        """

        super().__init__(**kwargs)

        if embed_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, but is {embed_dim}.")

        self.scale = scale
        self.embed_dim = embed_dim
        self.include_identity = include_identity
        self.initializer = initializer
        self.trainable = trainable
        self.w = None

    def call(self, t: Tensor) -> Tensor:
        """Embeds a time vector into a higher-dimensional Fourier embedding.

        Parameters
        ----------
        t   : Tensor of shape (batch_size, D)
            vector of times

        Returns
        -------
        emb : Tensor
            Embedding of shape (batch_size, D * embed_dim) if `include_identity`
            is False, else (batch_size, D * (embed_dim + 1))
        """
        # (batch_size, D, embed_dim // 2)
        proj = ops.expand_dims(t, axis=-1) * ops.reshape(self.w, (-1, self.embed_dim // 2)) * 2 * np.pi * self.scale
        emb = ops.concatenate([ops.sin(proj), ops.cos(proj)], axis=-1)
        emb = ops.reshape(emb, ops.shape(t)[:-1] + (-1,))
        if self.include_identity:
            emb = ops.concatenate([t, emb], axis=-1)
        return emb

    def build(self, input_shape):
        if self.built:
            return

        # Create the frequency weights
        dim = input_shape[-1] or 1
        shape = (self.embed_dim // 2,) if dim == 1 else (dim, self.embed_dim // 2)
        self.w = self.add_weight(
            name="frequencies", shape=shape, initializer=self.initializer, trainable=self.trainable
        )

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        dim = input_shape[-1] or 1
        if self.include_identity:
            output_dim = dim * (self.embed_dim + 1)
        else:
            output_dim = dim * self.embed_dim
        return tuple(input_shape[:-1]) + (output_dim,)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "embed_dim": self.embed_dim,
            "scale": self.scale,
            "initializer": self.initializer,
            "trainable": self.trainable,
            "include_identity": self.include_identity,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
