import numpy as np

import keras
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class FourierEmbedding(keras.Layer):
    """Implements a Fourier projection with normally distributed frequencies."""

    def __init__(
        self,
        embed_dim: int = 8,
        scale: float = 1.0,
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
            Dimensionality of the Fourier projection. The resulting embedding
            has dimensionality `embed_dim + 1` if `include_identity` is set to True.
        scale            : float, optional (default - 1.0)
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

        self.w = self.add_weight(initializer=initializer, shape=(embed_dim // 2,), trainable=trainable)
        self.scale = scale
        self.embed_dim = embed_dim
        self.include_identity = include_identity
        self.initializer = initializer
        self.trainable = trainable

    def call(self, t: Tensor) -> Tensor:
        """Embeds the one-dimensional time scalar into a higher-dimensional Fourier embedding.

        Parameters
        ----------
        t   : Tensor of shape (batch_size, 1)
            vector of times

        Returns
        -------
        emb : Tensor
            Embedding of shape (batch_size, fourier_emb_dim) if `include_identity`
            is False, else (batch_size, fourier_emb_dim+1)
        """
        proj = t * self.w[None, :] * 2 * np.pi * self.scale
        if self.include_identity:
            emb = ops.concatenate([t, ops.sin(proj), ops.cos(proj)], axis=-1)
        else:
            emb = ops.concatenate([ops.sin(proj), ops.cos(proj)], axis=-1)
        return emb

    def compute_output_shape(self, input_shape):
        if self.include_identity:
            return input_shape[:-1], self.embed_dim + 1
        else:
            return input_shape[:-1], self.embed_dim

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
