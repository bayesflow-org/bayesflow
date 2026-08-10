from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import expand_singletons_to_common_length, find_recurrent_net, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class RecurrentDecoder(keras.Layer):
    """Recurrent decoder for autoregressive target sequences.

    Stacks GRU/LSTM-gated recurrent layers over shifted targets and encoded
    summaries. In ``AutoregressiveApproximator``, it creates one condition per
    target time step for teacher forcing and cached autoregressive sampling.

    Parameters
    ----------
    embed_dim : int or sequence of int, optional
        Hidden units for each recurrent layer, by default 256.
    recurrent_type : str or sequence of str, optional
        Recurrent layer type, for example ``"gru"`` or ``"lstm"``, by default ``"gru"``.
    include_condition : bool, optional
        Whether to concatenate the matching encoder output to each decoded condition,
        by default True.
    output_dim : int or None, optional
        Dimensionality of the projected decoder output. If None, uses the final
        recurrent embedding dimension.
    layer_norm : bool, optional
        Whether to apply RMSNorm before the output projection, by default True.
    """

    def __init__(
        self,
        embed_dim: int | Sequence[int] = 256,
        recurrent_type: str | Sequence[str] = "gru",
        include_condition: bool = True,
        output_dim: int | None = None,
        layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        recurrent_kwargs = expand_singletons_to_common_length(
            embed_dim=embed_dim,
            recurrent_type=recurrent_type,
        )

        self.embed_dims = recurrent_kwargs["embed_dim"]
        self.output_dim = output_dim if output_dim is not None else self.embed_dims[-1]
        self.recurrent_layers = [
            find_recurrent_net(rnn_type, units=embed, return_sequences=True, return_state=True)
            for embed, rnn_type in zip(
                recurrent_kwargs["embed_dim"],
                recurrent_kwargs["recurrent_type"],
                strict=True,
            )
        ]

        self.embed_dim = embed_dim
        self.recurrent_type = recurrent_type
        self.include_condition = include_condition
        self.layer_norm = layer_norm
        self.output_norm = keras.layers.RMSNormalization(axis=-1) if layer_norm else None
        self.output_projection = keras.layers.Dense(self.output_dim, use_bias=False)
        self.bos_embedding = None

    def build(self, inference_variables_shape, encoder_outputs_shape):
        if self.built:
            return

        target_dim = inference_variables_shape[-1]
        self.bos_embedding = self.add_weight(
            name="bos_embedding",
            shape=(1, 1, target_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        recurrent_input_shape = tuple(inference_variables_shape[:-1]) + (
            inference_variables_shape[-1] + encoder_outputs_shape[-1],
        )
        for recurrent_layer, embed in zip(self.recurrent_layers, self.embed_dims, strict=True):
            recurrent_layer.build(recurrent_input_shape)
            recurrent_input_shape = tuple(recurrent_input_shape[:-1]) + (embed,)

        if self.output_norm is not None:
            self.output_norm.build(recurrent_input_shape)
        self.output_projection.build(recurrent_input_shape)

    def call(
        self,
        inference_variables: Tensor,
        encoder_outputs: Tensor,
        *,
        target_mask: Tensor | None = None,
        encoder_mask: Tensor | None = None,
        training: bool = False,
    ) -> Tensor:
        encoder_outputs = self._mask_encoder_outputs(encoder_outputs, encoder_mask)
        shifted_targets = self._shift_targets_with_bos(inference_variables, self.bos_embedding, target_mask)

        memory = keras.ops.concatenate([shifted_targets, encoder_outputs], axis=-1)
        for recurrent_layer in self.recurrent_layers:
            memory, *_ = recurrent_layer(memory, training=training)

        if self.output_norm is not None:
            memory = self.output_norm(memory, training=training)
        memory = self.output_projection(memory)

        if self.include_condition:
            memory = keras.ops.concatenate([memory, encoder_outputs], axis=-1)

        return memory

    def initialize_cache(self, encoder_outputs: Tensor, encoder_mask: Tensor | None = None) -> dict:
        return {
            "encoder_outputs": encoder_outputs,
            "encoder_mask": encoder_mask,
            "states": [None] * len(self.recurrent_layers),
        }

    def decode_step(
        self,
        previous_target: Tensor | None,
        *,
        step: int,
        cache: dict,
        target_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        encoder_outputs = cache["encoder_outputs"]
        batch_size = keras.ops.shape(encoder_outputs)[0]

        if previous_target is None:
            previous_target = keras.ops.broadcast_to(
                self.bos_embedding,
                (batch_size, *keras.ops.shape(self.bos_embedding)[1:]),
            )
        else:
            if target_mask is not None and step > 0:
                previous_target = previous_target * keras.ops.cast(
                    target_mask[:, step - 1 : step],
                    previous_target.dtype,
                )
            previous_target = previous_target[:, None, :]

        step_condition = encoder_outputs[:, step : step + 1]

        step_condition = self._mask_encoder_outputs(
            step_condition,
            None if cache.get("encoder_mask") is None else cache["encoder_mask"][:, step : step + 1],
        )

        condition = keras.ops.concatenate([previous_target, step_condition], axis=-1)
        new_states = []
        for recurrent_layer, state in zip(self.recurrent_layers, cache["states"], strict=True):
            recurrent_kwargs = {}
            if state is not None:
                recurrent_kwargs["initial_state"] = state

            result = recurrent_layer(condition, **recurrent_kwargs)
            condition = result[0]
            new_states.append(tuple(result[1:]))

        if self.output_norm is not None:
            condition = self.output_norm(condition, training=False)
        condition = self.output_projection(condition)

        if self.include_condition:
            condition = keras.ops.concatenate([condition, step_condition], axis=-1)

        return condition[:, 0], cache | {"states": new_states}

    def compute_output_shape(self, inference_variables_shape, encoder_outputs_shape):
        output_dim = self.output_dim
        if self.include_condition:
            output_dim += encoder_outputs_shape[-1]
        return tuple(inference_variables_shape[:-1]) + (output_dim,)

    def get_config(self):
        return super().get_config() | serialize(
            {
                "embed_dim": self.embed_dim,
                "recurrent_type": self.recurrent_type,
                "include_condition": self.include_condition,
                "output_dim": self.output_dim,
                "layer_norm": self.layer_norm,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = deserialize(config, custom_objects=custom_objects)
        if "hidden_dim" in config and "embed_dim" not in config:
            config["embed_dim"] = config.pop("hidden_dim")
        if "summary_dim" in config and "output_dim" not in config:
            config["output_dim"] = config.pop("summary_dim")
        return cls(**config)

    @staticmethod
    def _mask_encoder_outputs(encoder_outputs: Tensor, encoder_mask: Tensor | None) -> Tensor:
        if encoder_mask is None:
            return encoder_outputs

        return encoder_outputs * keras.ops.cast(encoder_mask[..., None], encoder_outputs.dtype)

    @staticmethod
    def _shift_targets_with_bos(
        inference_variables: Tensor,
        bos_embedding: Tensor,
        target_mask: Tensor | None,
    ) -> Tensor:
        batch_size = keras.ops.shape(inference_variables)[0]
        bos = keras.ops.broadcast_to(
            bos_embedding,
            (batch_size, 1, keras.ops.shape(inference_variables)[-1]),
        )
        shifted_targets = keras.ops.concatenate([bos, inference_variables[:, :-1, :]], axis=1)
        if target_mask is None:
            return shifted_targets

        shifted_mask = keras.ops.concatenate(
            [keras.ops.ones((batch_size, 1), dtype=target_mask.dtype), target_mask[:, :-1]],
            axis=1,
        )
        return shifted_targets * keras.ops.cast(shifted_mask[..., None], shifted_targets.dtype)
