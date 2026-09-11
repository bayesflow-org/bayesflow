from collections.abc import Sequence

import keras

from bayesflow.networks.helpers import Time2Vec
from bayesflow.types import Tensor
from bayesflow.utils import expand_singletons_to_common_length, find_recurrent_net, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from .. import SummaryNetwork


@serializable("bayesflow.networks")
class RecurrentNetwork(SummaryNetwork):
    """Recurrent summary network for time series.

    Stacks GRU/LSTM-style recurrent layers on top of optional Time2Vec embeddings.
    By default, the network acts as a many-to-one summary network and returns a
    single vector of size ``summary_dim``. If ``return_sequences=True``, the
    final recurrent layer keeps its full sequence output and the network returns a
    sequence of summary vectors.

    Parameters
    ----------
    summary_dim : int, optional
        Dimensionality of the projected summary output, by default 64.
    hidden_dim : int or sequence of int, optional
        Hidden units for each recurrent layer, by default ``(128, 128)``.
    recurrent_type : str or sequence of str, optional
        Recurrent layer type, for example ``"gru"`` or ``"lstm"``, by default ``"gru"``.
    bidirectional : bool or sequence of bool, optional
        Whether to wrap recurrent layers in ``Bidirectional``, by default True.
    merge_mode : str or sequence of str, optional
        Merge mode for bidirectional layers, by default ``"sum"``.
    time_axis : int or None, optional
        Feature axis containing explicit time values. If None, integer positions
        are used by the time embedding.
    time_embed_dim : int, optional
        Number of time embedding features, by default 16.
    dropout : float, optional
        Dropout rate after the recurrent stack, by default 0.05.
    kernel_initializer : str, optional
        Initializer for recurrent input kernels and the output projection, by
        default ``"orthogonal"``.
    return_sequences : bool, optional
        Whether to return one summary per time step. If False, returns only the
        final recurrent output projected to ``summary_dim``.
    """

    def __init__(
        self,
        summary_dim: int = 16,
        hidden_dim: int | Sequence[int] = (128, 128),
        recurrent_type: str | Sequence[str] = "gru",
        bidirectional: bool | Sequence[bool] = True,
        merge_mode: str | Sequence[str] = "sum",
        time_axis: int | None = None,
        time_embed_dim: int = 16,
        dropout: float = 0.05,
        kernel_initializer: str = "orthogonal",
        return_sequences: bool = False,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        recurrent_kwargs = expand_singletons_to_common_length(
            hidden_dim=hidden_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
        )

        self.recurrent_layers = []

        num_layers = len(recurrent_kwargs["hidden_dim"])
        return_sequences_by_layer = (True,) * (num_layers - 1) + (return_sequences,)

        for recurrent_layer_kwargs in zip(
            *recurrent_kwargs.values(),
            return_sequences_by_layer,
            strict=True,
        ):
            hidden, rnn_type, bidir, merge, layer_return_sequences = recurrent_layer_kwargs
            recurrent_layer = find_recurrent_net(
                rnn_type,
                units=hidden,
                return_sequences=layer_return_sequences,
                kernel_initializer=kernel_initializer,
            )

            if bidir:
                recurrent_layer = keras.layers.Bidirectional(recurrent_layer, merge_mode=merge)

            self.recurrent_layers.append(recurrent_layer)

        self.dropout_layer = keras.layers.Dropout(dropout)
        self.time_embedding = Time2Vec(num_periodic_features=time_embed_dim - 1)

        self.output_projector = keras.layers.Dense(
            summary_dim,
            kernel_initializer=kernel_initializer,
        )

        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode
        self.time_axis = time_axis
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.return_sequences = return_sequences

    def build(self, time_series_shape):
        if self.built:
            return

        input_dim = time_series_shape[-1] - 1 if self.time_axis is not None else time_series_shape[-1]
        recurrent_shape = tuple(time_series_shape[:-1]) + (input_dim + self.time_embed_dim,)
        self.time_embedding.build(tuple(time_series_shape[:-1]) + (input_dim,))

        for recurrent_layer in self.recurrent_layers:
            recurrent_layer.build(recurrent_shape)
            recurrent_shape = recurrent_layer.compute_output_shape(recurrent_shape)

        self.output_projector.build(recurrent_shape)

    def call(self, time_series: Tensor, training: bool = False, mask: Tensor | None = None, **kwargs) -> Tensor:
        out, time_vec = self._split_time_series(time_series)
        out = self.time_embedding(out, t=time_vec)

        for recurrent_layer in self.recurrent_layers:
            out = recurrent_layer(out, training=training, mask=mask)

        out = self.dropout_layer(out, training=training)
        return self.output_projector(out)

    def compute_output_shape(self, time_series_shape):
        if self.return_sequences:
            return tuple(time_series_shape[:-1]) + (self.summary_dim,)
        return tuple(time_series_shape[:-2]) + (self.summary_dim,)

    def get_config(self):
        return super().get_config() | serialize(
            {
                "summary_dim": self.summary_dim,
                "hidden_dim": self.hidden_dim,
                "recurrent_type": self.recurrent_type,
                "bidirectional": self.bidirectional,
                "merge_mode": self.merge_mode,
                "dropout": self.dropout,
                "time_axis": self.time_axis,
                "time_embed_dim": self.time_embed_dim,
                "kernel_initializer": self.kernel_initializer,
                "return_sequences": self.return_sequences,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def _split_time_series(self, time_series: Tensor) -> tuple[Tensor, Tensor | None]:
        if self.time_axis is None:
            return time_series, None

        num_features = time_series.shape[-1]
        time_axis = self.time_axis if self.time_axis >= 0 else num_features + self.time_axis
        time = time_series[..., time_axis]
        indices = list(range(num_features))
        indices.pop(time_axis)
        return keras.ops.take(time_series, indices, axis=-1), time
