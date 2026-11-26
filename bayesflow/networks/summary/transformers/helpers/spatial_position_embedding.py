import math

import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.networks")
class SpatialPositionEmbedding(keras.Layer):
    """Add deterministic sinusoidal row and column positions to a feature grid.

    The input is expected to have shape ``(batch, time, height, width,
    embed_dim)``. Row and column encodings are computed independently and
    broadcast over the batch and time axes. Complementary sine/cosine phases
    distinguish the two spatial axes without imposing divisibility constraints
    on ``embed_dim``.

    Parameters
    ----------
    temperature : float, optional
        Geometric frequency scale of the sinusoidal encoding, by default
        10,000.
    **kwargs
        Additional keyword arguments passed to ``keras.Layer``.
    """

    def __init__(self, temperature: float = 10_000.0, **kwargs):
        super().__init__(**kwargs)

        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature!r}.")

        self.temperature = float(temperature)

    def call(self, x: Tensor) -> Tensor:
        _, _, height, width, embed_dim = keras.ops.shape(x)

        input_dtype = keras.backend.standardize_dtype(x.dtype)
        position_dtype = "float64" if input_dtype == "float64" else "float32"

        row_embedding = self._axis_embedding(
            height,
            embed_dim,
            dtype=position_dtype,
            swap_sin_cos=False,
        )
        column_embedding = self._axis_embedding(
            width,
            embed_dim,
            dtype=position_dtype,
            swap_sin_cos=True,
        )

        row_embedding = keras.ops.reshape(row_embedding, (1, 1, height, 1, embed_dim))
        column_embedding = keras.ops.reshape(column_embedding, (1, 1, 1, width, embed_dim))
        position_embedding = keras.ops.cast(row_embedding + column_embedding, x.dtype)

        return x + position_embedding

    def _axis_embedding(
        self,
        length: int,
        embed_dim: int,
        *,
        dtype: str,
        swap_sin_cos: bool,
    ) -> Tensor:
        positions = keras.ops.cast(keras.ops.arange(length), dtype)
        dimensions = keras.ops.arange(embed_dim)
        frequency_indices = keras.ops.floor(keras.ops.cast(dimensions, dtype) / 2.0)
        exponent = 2.0 * frequency_indices / keras.ops.cast(embed_dim, dtype)
        inverse_frequencies = keras.ops.exp(-math.log(self.temperature) * exponent)
        angles = positions[:, None] * inverse_frequencies[None, :]

        even_dimensions = dimensions % 2 == 0
        sine = keras.ops.sin(angles)
        cosine = keras.ops.cos(angles)
        if swap_sin_cos:
            return keras.ops.where(even_dimensions[None, :], cosine, sine)
        return keras.ops.where(even_dimensions[None, :], sine, cosine)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize({"temperature": self.temperature})
