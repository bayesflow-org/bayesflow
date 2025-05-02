import keras

from bayesflow.types import Tensor
from bayesflow.utils import expand_tile

from bayesflow.utils.serialization import serializable


@serializable("bayesflow.networks")
class RecurrentEmbedding(keras.Layer):
    """Implements a recurrent network for flexibly embedding time vectors."""

    def __init__(self, embed_dim: int = 8, embedding: str = "lstm"):
        super().__init__()

        self.embed_dim = embed_dim
        self.embedding = embedding
        if embedding == "lstm":
            self.embedder = keras.layers.LSTM(embed_dim, return_sequences=True)
        elif embedding == "gru":
            self.embedder = keras.layers.GRU(embed_dim, return_sequences=True)
        else:
            raise ValueError(f"Unknown embedding type {embedding}. Must be in ['lstm', 'gru']")

    def call(self, x: Tensor, t: Tensor = None) -> Tensor:
        """Creates time representations and concatenates them to x.

        Parameters
        ----------
        x   : Tensor of shape (batch_size, sequence_length, dim)
            The input sequence.
        t   : Tensor of shape (batch_size, sequence_length)
            Vector of times

        Returns
        -------
        emb : Tensor
            Embedding of shape (batch_size, sequence_length, embed_dim + 1)
        """

        if t is None:
            t = keras.ops.linspace(0, keras.ops.shape(x)[1], keras.ops.shape(x)[1], dtype=x.dtype)
            t = expand_tile(t, keras.ops.shape(x)[0], axis=0)

        emb = self.embedder(t[..., None])
        return keras.ops.concatenate([x, emb], axis=-1)
