"""Recurrent sequence networks."""

from collections.abc import Sequence

import keras

from bayesflow.networks.helpers import Time2Vec
from bayesflow.types import Tensor
from bayesflow.utils import expand_singletons_to_common_length, find_recurrent_net, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ...summary import SummaryNetwork


@serializable("bayesflow.networks")
class MemoryDecoder(keras.Layer):
    """Minimal recurrent memory decoder for autoregressive approximators."""

    def __init__(
        self,
        hidden_dim: int = 256,
        recurrent_type: str = "gru",
        include_condition: bool = True,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.recurrent = find_recurrent_net(recurrent_type)(units=hidden_dim, return_sequences=True, return_state=True)
        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.include_condition = include_condition
        self.target_dim = None
        self.bos_embedding = None

    def build(self, inference_variables_shape, encoder_outputs_shape):
        if self.built:
            return

        self.target_dim = inference_variables_shape[-1]
        self.bos_embedding = self.add_weight(
            name="bos_embedding",
            shape=(1, 1, self.target_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        recurrent_input_shape = tuple(inference_variables_shape[:-1]) + (
            inference_variables_shape[-1] + encoder_outputs_shape[-1],
        )
        self.recurrent.build(recurrent_input_shape)

    def call(self, inference_variables: Tensor, encoder_outputs: Tensor, training: bool = False, **kwargs) -> Tensor:
        shifted_targets = self._shift_targets_with_bos(inference_variables, self.bos_embedding)
        memory, *_ = self.recurrent(
            keras.ops.concatenate([shifted_targets, encoder_outputs], axis=-1),
            training=training,
        )
        if self.include_condition:
            return keras.ops.concatenate([memory, encoder_outputs], axis=-1)
        return memory

    def initialize_cache(self, encoder_outputs: Tensor, encoder_mask: Tensor | None = None, **kwargs) -> dict:
        return {"encoder_outputs": encoder_outputs, "encoder_mask": encoder_mask, "state": None}

    def decode_step(
        self,
        previous_target: Tensor | None,
        *,
        step: int,
        cache: dict,
        **kwargs,
    ) -> tuple[Tensor, dict]:
        encoder_outputs = cache["encoder_outputs"]
        batch_size = keras.ops.shape(encoder_outputs)[0]

        if previous_target is None:
            previous_target = keras.ops.broadcast_to(
                self.bos_embedding,
                (batch_size, 1, self.target_dim),
            )
        else:
            previous_target = previous_target[:, None, :]

        step_condition = encoder_outputs[:, step : step + 1]

        if cache.get("encoder_mask") is not None:
            step_condition = step_condition * keras.ops.cast(
                cache["encoder_mask"][:, step : step + 1, None],
                step_condition.dtype,
            )

        recurrent_kwargs = {}
        if cache["state"] is not None:
            recurrent_kwargs["initial_state"] = cache["state"]

        result = self.recurrent(
            keras.ops.concatenate([previous_target, step_condition], axis=-1),
            **recurrent_kwargs,
        )
        condition = result[0]
        if self.include_condition:
            condition = keras.ops.concatenate([condition, step_condition], axis=-1)

        return condition[:, 0], cache | {"state": tuple(result[1:])}

    def compute_output_shape(self, inference_variables_shape, encoder_outputs_shape):
        output_dim = self.hidden_dim + encoder_outputs_shape[-1] if self.include_condition else self.hidden_dim
        return tuple(inference_variables_shape[:-1]) + (output_dim,)

    def get_config(self):
        return super().get_config() | serialize(
            {
                "hidden_dim": self.hidden_dim,
                "recurrent_type": self.recurrent_type,
                "include_condition": self.include_condition,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    @staticmethod
    def _shift_targets_with_bos(inference_variables: Tensor, bos_embedding: Tensor) -> Tensor:
        batch_size = keras.ops.shape(inference_variables)[0]
        bos = keras.ops.broadcast_to(
            bos_embedding,
            (batch_size, 1, keras.ops.shape(inference_variables)[-1]),
        )
        return keras.ops.concatenate([bos, inference_variables[:, :-1, :]], axis=1)


@serializable("bayesflow.networks")
class RecurrentNet(SummaryNetwork):
    """Sequence-to-sequence recurrent summary network."""

    def __init__(
        self,
        summary_dim: int = 64,
        hidden_dim: int | Sequence[int] = (128, 128),
        recurrent_type: str | Sequence[str] = "gru",
        bidirectional: bool | Sequence[bool] = True,
        merge_mode: str | Sequence[str] = "sum",
        layer_norm: bool | Sequence[bool] = True,
        time_axis: int | None = 0,
        time_embed_dim: int = 16,
        dropout: float = 0.05,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        recurrent_kwargs = expand_singletons_to_common_length(
            hidden_dim=hidden_dim,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            merge_mode=merge_mode,
            layer_norm=layer_norm,
        )

        self.recurrent_layers = []
        self.normalization_layers = []

        for hidden, rnn_type, bidir, merge, norm in zip(*recurrent_kwargs.values(), strict=True):
            recurrent_layer = find_recurrent_net(rnn_type)(units=hidden, return_sequences=True)

            if bidir:
                recurrent_layer = keras.layers.Bidirectional(recurrent_layer, merge_mode=merge)

            self.recurrent_layers.append(recurrent_layer)
            self.normalization_layers.append(keras.layers.LayerNormalization() if norm else None)

        self.dropout_layer = keras.layers.Dropout(dropout)
        self.summary_stats = keras.layers.Conv1D(filters=summary_dim, kernel_size=1)
        self.time_embedding = Time2Vec(num_periodic_features=time_embed_dim - 1)

        self.summary_dim = summary_dim
        self.hidden_dim = hidden_dim
        self.recurrent_type = recurrent_type
        self.bidirectional = bidirectional
        self.merge_mode = merge_mode
        self.layer_norm = layer_norm
        self.time_axis = time_axis
        self.time_embed_dim = time_embed_dim
        self.dropout = dropout

    def build(self, time_series_shape):
        if self.built:
            return

        input_dim = time_series_shape[-1] - 1 if self.time_axis is not None else time_series_shape[-1]
        recurrent_shape = tuple(time_series_shape[:-1]) + (input_dim + self.time_embed_dim,)
        self.time_embedding.build(tuple(time_series_shape[:-1]) + (input_dim,))

        for recurrent_layer, normalization_layer in zip(
            self.recurrent_layers,
            self.normalization_layers,
            strict=True,
        ):
            recurrent_layer.build(recurrent_shape)
            recurrent_shape = recurrent_layer.compute_output_shape(recurrent_shape)
            if normalization_layer is not None:
                normalization_layer.build(recurrent_shape)

        self.summary_stats.build(recurrent_shape)

    def call(self, time_series: Tensor, training: bool = False, mask: Tensor | None = None, **kwargs) -> Tensor:
        out, time = self._split_time_series(time_series)
        out = self.time_embedding(out, t=time)

        for recurrent_layer, normalization_layer in zip(
            self.recurrent_layers,
            self.normalization_layers,
            strict=True,
        ):
            out = recurrent_layer(out, training=training, mask=mask)
            if normalization_layer is not None:
                out = normalization_layer(out, training=training)

        out = self.dropout_layer(out, training=training)
        return self.summary_stats(out)

    def compute_output_shape(self, time_series_shape):
        return tuple(time_series_shape[:-1]) + (self.summary_dim,)

    def get_config(self):
        return super().get_config() | serialize(
            {
                "summary_dim": self.summary_dim,
                "hidden_dim": self.hidden_dim,
                "recurrent_type": self.recurrent_type,
                "bidirectional": self.bidirectional,
                "merge_mode": self.merge_mode,
                "layer_norm": self.layer_norm,
                "dropout": self.dropout,
                "time_axis": self.time_axis,
                "time_embed_dim": self.time_embed_dim,
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
