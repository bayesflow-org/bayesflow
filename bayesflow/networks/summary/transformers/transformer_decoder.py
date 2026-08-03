import keras

from bayesflow.networks.helpers import Time2Vec
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from .attention import MultiHeadAttention


@serializable("bayesflow.networks")
class TransformerDecoder(keras.Layer):
    """Causal transformer decoder with incremental attention caching.

    Each layer alternates the existing BayesFlow ``MultiHeadAttention``
    block between causal self-attention over shifted targets and unrestricted
    cross-attention over the complete encoded observation sequence.
    """

    def __init__(
        self,
        summary_dim: int = 32,
        embed_dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        use_bias: bool = False,
        layer_norm: bool = True,
        time_embed_dim: int = 8,
        kernel_initializer: str = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.summary_dim = summary_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.use_bias = use_bias
        self.layer_norm = layer_norm
        self.time_embed_dim = time_embed_dim
        self.kernel_initializer = kernel_initializer

        attention_kwargs = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "dropout": dropout,
            "expansion_factor": expansion_factor,
            "glu_variant": glu_variant,
            "use_bias": use_bias,
            "layer_norm": layer_norm,
            "kernel_initializer": kernel_initializer,
        }
        self.self_attention_blocks = [MultiHeadAttention(**attention_kwargs) for _ in range(num_layers)]
        self.cross_attention_blocks = [MultiHeadAttention(**attention_kwargs) for _ in range(num_layers)]

        self.time_embedding = Time2Vec(num_periodic_features=time_embed_dim - 1)
        self.target_projection = keras.layers.Dense(
            embed_dim,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )
        self.output_norm = keras.layers.RMSNormalization(axis=-1) if layer_norm else None
        self.output_projection = keras.layers.Dense(
            summary_dim,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )
        self.bos_embedding = None

    def build(self, inference_variables_shape, encoder_outputs_shape):
        if self.built:
            return

        target_dim = inference_variables_shape[-1]
        embedded_shape = tuple(inference_variables_shape[:-1]) + (target_dim + self.time_embed_dim,)
        if not self.time_embedding.built:
            self.time_embedding.build(inference_variables_shape)
        self.target_projection.build(embedded_shape)

        decoder_shape = self.target_projection.compute_output_shape(embedded_shape)

        self.bos_embedding = self.add_weight(
            name="bos_embedding",
            shape=(1, 1, target_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
        )

        for self_attention, cross_attention in zip(
            self.self_attention_blocks,
            self.cross_attention_blocks,
            strict=True,
        ):
            self_attention.build(decoder_shape, decoder_shape)
            cross_attention.build(decoder_shape, encoder_outputs_shape)

        if self.output_norm is not None:
            self.output_norm.build(decoder_shape)

        self.output_projection.build(decoder_shape)

    def call(
        self,
        inference_variables: Tensor,
        encoder_outputs: Tensor,
        *,
        target_mask: Tensor | None = None,
        encoder_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        training: bool = False,
    ) -> Tensor:
        """Create all sequence conditions in parallel using teacher forcing."""
        batch_size, num_steps = keras.ops.shape(inference_variables)[:2]
        bos = keras.ops.broadcast_to(
            self.bos_embedding,
            (batch_size, 1, keras.ops.shape(inference_variables)[-1]),
        )
        shifted_targets = keras.ops.concatenate([bos, inference_variables[:, :-1]], axis=1)
        x = self.target_projection(self.time_embedding(shifted_targets))

        self_attention_mask = self._shift_attention_mask(
            target_mask,
            attention_mask,
            batch_size,
            num_steps,
        )
        encoder_attention_mask = None if encoder_mask is None else keras.ops.cast(encoder_mask[:, None, :], "bool")

        for self_attention, cross_attention in zip(
            self.self_attention_blocks,
            self.cross_attention_blocks,
            strict=True,
        ):
            x = self_attention(
                x,
                x,
                training=training,
                attention_mask=self_attention_mask,
                use_causal_mask=True,
            )
            x = cross_attention(
                x,
                encoder_outputs,
                training=training,
                attention_mask=encoder_attention_mask,
            )

        if self.output_norm is not None:
            x = self.output_norm(x, training=training)

        return self.output_projection(x)

    def initialize_cache(self, encoder_outputs: Tensor, encoder_mask: Tensor | None = None) -> dict:
        """Cache projected encoder keys and values for every cross-attention block."""
        encoder_attention_mask = None if encoder_mask is None else keras.ops.cast(encoder_mask[:, None, :], "bool")
        return {
            "cross_key_values": [
                attention.prepare_key_value(encoder_outputs, training=False)
                for attention in self.cross_attention_blocks
            ],
            "self_key_values": [None] * self.num_layers,
            "encoder_attention_mask": encoder_attention_mask,
        }

    def decode_step(
        self,
        previous_target: Tensor | None,
        *,
        step: int,
        cache: dict,
        target_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Decode one time point and append one entry to each self-attention cache."""
        batch_size = keras.ops.shape(cache["cross_key_values"][0][0])[0]
        if previous_target is None:
            shifted_target = keras.ops.broadcast_to(
                self.bos_embedding,
                (batch_size, 1, keras.ops.shape(self.bos_embedding)[-1]),
            )
        else:
            shifted_target = previous_target[:, None, :]

        time = keras.ops.full((batch_size, 1), step, dtype=shifted_target.dtype)
        x = self.target_projection(self.time_embedding(shifted_target, t=time))

        self_attention_mask = None
        if target_mask is not None:
            self_attention_mask = keras.ops.concatenate(
                [
                    keras.ops.ones((batch_size, 1), dtype="bool"),
                    keras.ops.cast(target_mask[:, :step], "bool"),
                ],
                axis=1,
            )[:, None, :]
        if attention_mask is not None:
            step_attention = keras.ops.concatenate(
                [
                    keras.ops.ones((batch_size, 1), dtype="bool"),
                    keras.ops.cast(attention_mask[:, step, :step], "bool"),
                ],
                axis=1,
            )[:, None, :]
            self_attention_mask = (
                step_attention
                if self_attention_mask is None
                else keras.ops.logical_and(self_attention_mask, step_attention)
            )

        new_self_key_values = []
        for index, (self_attention, cross_attention) in enumerate(
            zip(self.self_attention_blocks, self.cross_attention_blocks, strict=True)
        ):
            x, self_key_value = self_attention.call_with_cache(
                x,
                y=x,
                key_value_cache=cache["self_key_values"][index],
                append_key_value=True,
                attention_mask=self_attention_mask,
            )
            x, _ = cross_attention.call_with_cache(
                x,
                key_value_cache=cache["cross_key_values"][index],
                attention_mask=cache["encoder_attention_mask"],
            )
            new_self_key_values.append(self_key_value)

        if self.output_norm is not None:
            x = self.output_norm(x, training=False)
        condition = self.output_projection(x)[:, 0]
        return condition, cache | {"self_key_values": new_self_key_values}

    @staticmethod
    def _shift_attention_mask(
        target_mask: Tensor | None,
        attention_mask: Tensor | None,
        batch_size: int,
        num_steps: int,
    ) -> Tensor | None:
        mask = None
        if target_mask is not None:
            target_mask = keras.ops.cast(target_mask, "bool")
            mask = keras.ops.concatenate(
                [keras.ops.ones((batch_size, 1), dtype="bool"), target_mask[:, :-1]],
                axis=1,
            )[:, None, :]

        if attention_mask is not None:
            attention_mask = keras.ops.cast(attention_mask, "bool")
            shifted_attention = keras.ops.concatenate(
                [
                    keras.ops.ones((batch_size, num_steps, 1), dtype="bool"),
                    attention_mask[:, :, :-1],
                ],
                axis=2,
            )
            mask = shifted_attention if mask is None else keras.ops.logical_and(mask, shifted_attention)
        return mask

    def compute_output_shape(self, inference_variables_shape, encoder_outputs_shape):
        return tuple(inference_variables_shape)[:-1] + (self.summary_dim,)

    def get_config(self):
        return super().get_config() | serialize(
            {
                "summary_dim": self.summary_dim,
                "embed_dim": self.embed_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "dropout": self.dropout_rate,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "use_bias": self.use_bias,
                "layer_norm": self.layer_norm,
                "time_embed_dim": self.time_embed_dim,
                "kernel_initializer": self.kernel_initializer,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
