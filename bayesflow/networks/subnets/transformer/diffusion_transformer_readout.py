import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ...helpers import DiffusionTransformerBlock


@serializable("bayesflow.networks")
class DiffusionTransformerReadout(keras.Layer):
    """Final layer of a :class:`~bayesflow.networks.DiffusionTransformer`: a time-conditioned output norm and a
    per-token projection, both zero-initialized. Called on ``(h, context)`` with the context returned by
    :meth:`~bayesflow.networks.DiffusionTransformer.embed`.

    Parameters
    ----------
    width : int
        Token embedding width.
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.
    """

    def __init__(self, width: int, **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        self.width = width
        self.out_norm = keras.layers.LayerNormalization(center=False, scale=False)
        self.final_ada_ln = keras.layers.Dense(2 * width, kernel_initializer="zeros", bias_initializer="zeros")
        self.token_out = keras.layers.Dense(1, kernel_initializer="zeros")

    def build(self, input_shape):
        if self.built:
            return

        h_shape, context_shape = input_shape
        self.out_norm.build(h_shape)
        self.final_ada_ln.build(context_shape[1])
        self.token_out.build(h_shape)

    def call(self, inputs: tuple[Tensor, tuple], training: bool | None = None) -> Tensor:
        h, context = inputs
        return self.final_layer(h, context, self.out_norm, self.final_ada_ln, self.token_out, training=training)

    @staticmethod
    def final_layer(
        h: Tensor,
        context: tuple,
        out_norm: keras.Layer,
        final_ada_ln: keras.Layer,
        token_out: keras.Layer,
        training: bool | None = None,
    ) -> Tensor:
        """The final layer computed with the given layers."""
        t_emb = context[1]
        shift, scale = keras.ops.split(keras.ops.expand_dims(final_ada_ln(t_emb), axis=1), 2, axis=-1)
        h = DiffusionTransformerBlock.modulate(out_norm(h, training=training), shift, scale)
        return keras.ops.squeeze(token_out(h), axis=-1)

    def compute_output_shape(self, input_shape):
        h_shape = input_shape[0]
        return tuple(h_shape[:-1])

    def get_config(self):
        base_config = layer_kwargs(super().get_config())
        return base_config | serialize({"width": self.width})

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
