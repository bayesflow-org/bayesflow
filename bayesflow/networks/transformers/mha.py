import math

import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor


@serializable(package="bayesflow.networks")
class MultiHeadAttention(keras.Layer):
    def __init__(self, embed_dim: int = 32, num_heads: int = 8, dropout: float = 0.05, use_bias: bool = True, **kwargs):
        """
        MultiHeadAttention layer that performs scaled dot-product attention.

        This layer projects queries, keys, and values into multiple heads,
        computes attention scores, and applies linear transformations on the outputs.

        Parameters
        ----------
        embed_dim : int, optional (default - 64)
            The dimensionality of the inner projection layers and output.
        num_heads : int, optional (default - 4)
            The number of attention heads.
        dropout   : float or None, optional (default - 0.05)
            Dropout rate to be applied to the attention scores.
        use_bias  : bool, optional (default - True)
            Whether to use bias in the dense layers.
        **kwargs  : dict
            Additional keyword arguments to be passed to the `keras.Layer` class.
        """

        super().__init__(**kwargs)

        if embed_dim % num_heads != 0:
            raise ValueError(f"embd_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim, use_bias=use_bias)
        self.key_dense = layers.Dense(embed_dim, use_bias=use_bias)
        self.value_dense = layers.Dense(embed_dim, use_bias=use_bias)
        self.combine_heads = layers.Dense(embed_dim, use_bias=use_bias)

        self.dropout = layers.Dropout(dropout) if dropout and dropout > 0.0 else None

    def call(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor:
        """
        Perform the forward pass of the MultiHeadAttention layer.

        Parameters
        ----------
        query     : Tensor
            Query tensor of shape (batch_size, seq_len_q, query_dim), where
            `seq_len_q` is the length of the query sequence.
        key       : Tensor
            Key tensor of shape (batch_size, seq_len_k, key_dim), where
            `seq_len_k` is the length of the key sequence.
        value     : Tensor
            Value tensor of shape (batch_size, seq_len_v, value_dim), where
            `seq_len_v` is the length of the value sequence.
        mask      : Tensor, optional (default - None)
            Float tensor of shape broadcastable to (batch_size, num_heads, seq_len_q, seq_len_k)
            to mask out specific positions in the attention computation (default is None).
        training  : bool, optional (default - False)
            Boolean flag indicating whether the layer should behave in training mode
            (e.g., applying dropout) or inference mode (default is False).
        **kwargs  : dict
            Additional keyword arguments.

        Returns
        -------
        output: Tensor
            Output tensor of shape (batch_size, seq_len_q, embed_dim).
        """

        batch_size = keras.ops.shape(query)[0]

        # Linear projections
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # Split heads
        query = self._split_heads(query, batch_size)
        key = self._split_heads(key, batch_size)
        value = self._split_heads(value, batch_size)

        # Scaled dot-product attention using einsum
        # Compute attention scores
        # Q: (batch_size, num_heads, seq_len_q, projection_dim)
        # K: (batch_size, num_heads, seq_len_k, projection_dim)
        # scores = Q @ K^T / sqrt(dk)
        scores = keras.ops.einsum("bhqd, bhkd -> bhqk", query, key)
        scores = scores / math.sqrt(self.projection_dim)

        if mask is not None:
            # Add the mask to the scores. Assuming mask is additive.
            scores += mask * -1e9

        # Apply softmax to get attention weights
        attention_weights = keras.ops.softmax(scores, axis=-1)
        if self.dropout is not None:
            attention_weights = self.dropout(attention_weights, training=training)

        # Compute the attention output
        # V: (batch_size, num_heads, seq_len_v, projection_dim)
        # Output: (batch_size, num_heads, seq_len_q, projection_dim)
        attention_output = keras.ops.einsum("bhqk, bhvd -> bhqd", attention_weights, value)

        # Transpose and reshape back to (batch_size, seq_len_q, embed_dim)
        attention_output = keras.ops.transpose(attention_output, axes=[0, 2, 1, 3])
        concat_attention = keras.ops.reshape(attention_output, newshape=(batch_size, -1, self.embed_dim))

        # Final linear projection
        output = self.combine_heads(concat_attention)

        return output

    def _split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """
        Splits the last dimension into (num_heads, projection_dim).
        Transpose the result to shape (batch_size, num_heads, seq_len, projection_dim).

        Args:
            x: Tensor with shape (batch_size, seq_len, embed_dim)
            batch_size: Integer representing the batch size.

        Returns:
            Tensor with shape (batch_size, num_heads, seq_len, projection_dim)
        """

        x = keras.ops.reshape(x, newshape=(batch_size, -1, self.num_heads, self.projection_dim))
        return keras.ops.transpose(x, axes=[0, 2, 1, 3])

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "projection_dim": self.projection_dim,
                "query_dense": keras.saving.serialize_keras_object(self.query_dense),
                "key_dense": keras.saving.serialize_keras_object(self.key_dense),
                "value_dense": keras.saving.serialize_keras_object(self.value_dense),
                "combine_heads": keras.saving.serialize_keras_object(self.combine_heads),
                "dropout": keras.saving.serialize_keras_object(self.dropout),
            }
        )
        return config
