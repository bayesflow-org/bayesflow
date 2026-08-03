import keras

from bayesflow.types import Tensor
from bayesflow.utils import find_recurrent_net, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class RecurrentDecoder(keras.Layer):
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
        self.recurrent.build(recurrent_input_shape)

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

        memory, *_ = self.recurrent(
            keras.ops.concatenate([shifted_targets, encoder_outputs], axis=-1),
            training=training,
        )

        if self.include_condition:
            return keras.ops.concatenate([memory, encoder_outputs], axis=-1)
        return memory

    def initialize_cache(self, encoder_outputs: Tensor, encoder_mask: Tensor | None = None) -> dict:
        return {"encoder_outputs": encoder_outputs, "encoder_mask": encoder_mask, "state": None}

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
