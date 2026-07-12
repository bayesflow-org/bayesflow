import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class FFN(keras.Layer):
    """Gated feedforward network (FFN) for modern transformer blocks.

    Implements a two-projection GLU-style feedforward layer where the gate branch
    is passed through an activation and then multiplied element-wise with the
    value branch before the output projection. Four variants are supported:

    - **SwiGLU** (default): gate = SiLU(x @ W_gate) — used in LLaMA, PaLM-2
    - **GEGLU**: gate = GELU(x @ W_gate) — used in T5 v1.1, PaLM
    - **ReGLU**: gate = ReLU(x @ W_gate)
    - **LiGLU**: gate = x @ W_gate (linear gate, no activation)

    The intermediate dimension is scaled by ``2/3 * expansion_factor`` relative to
    ``embed_dim`` so that the total parameter count stays comparable to a standard
    (non-gated) FFN with the same ``expansion_factor``.

    Parameters
    ----------
    embed_dim : int, optional
        Input and output dimensionality, by default 64.
    expansion_factor : float, optional
        Multiplier controlling the intermediate width before the 2/3 GLU
        correction is applied, by default 4.0.
    glu_variant : str, optional
        Which gated activation to use. One of ``"swiglu"``, ``"geglu"``,
        ``"reglu"``, or ``"liglu"``, by default ``"swiglu"``.
    use_bias : bool, optional
        Whether dense projections include a bias term, by default False.
    dropout : float, optional
        Dropout rate applied after gating and before the output projection,
        by default 0.0 (disabled).
    kernel_initializer: str, optional
        The initialization method for the three matrices, by default "glorot_uniform".
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.

    References
    ----------
    Noam Shazeer (2020). "GLU Variants Improve Transformer."
    https://arxiv.org/abs/2002.05202
    """

    GLU_ACTIVATIONS: dict[str, str | None] = {
        "swiglu": "silu",
        "geglu": "gelu",
        "reglu": "relu",
        "liglu": None,
    }

    def __init__(
        self,
        embed_dim: int = 64,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        use_bias: bool = False,
        dropout: float = 0.0,
        kernel_initializer: str = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        glu_variant = glu_variant.lower()
        if glu_variant not in self.GLU_ACTIVATIONS:
            raise ValueError(f"Unknown GLU variant '{glu_variant}'. Choose from: {list(self.GLU_ACTIVATIONS.keys())}.")

        self.embed_dim = embed_dim
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.use_bias = use_bias
        self.dropout_rate = dropout
        self.kernel_initializer = kernel_initializer

        # Scale intermediate dim so total params ≈ standard FFN with expansion_factor
        intermediate_dim = int(embed_dim * expansion_factor * 2 / 3)

        # Round up to nearest multiple of 64 for hardware efficiency
        intermediate_dim = ((intermediate_dim + 63) // 64) * 64
        self.intermediate_dim = intermediate_dim

        dense_kwargs = dict(use_bias=use_bias, kernel_initializer=kernel_initializer)

        self.gate_proj = layers.Dense(intermediate_dim, **dense_kwargs)
        self.up_proj = layers.Dense(intermediate_dim, **dense_kwargs)
        self.down_proj = layers.Dense(embed_dim, **dense_kwargs)

        self.dropout = layers.Dropout(dropout) if dropout > 0.0 else None

        activation_name = self.GLU_ACTIVATIONS[glu_variant]
        self.gate_fn = keras.activations.get(activation_name) if activation_name else None

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        gate = self.gate_proj(x)

        if self.gate_fn is not None:
            gate = self.gate_fn(gate)

        x = gate * self.up_proj(x)

        if self.dropout is not None:
            x = self.dropout(x, training=training)

        return self.down_proj(x)

    def build(self, input_shape):
        if self.built:
            return

        input_shape = input_shape
        self.gate_proj.build(input_shape)
        self.up_proj.build(input_shape)
        inter_shape = tuple(input_shape)[:-1] + (self.intermediate_dim,)
        self.down_proj.build(inter_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape)[:-1] + (self.embed_dim,)

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
                "embed_dim": self.embed_dim,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "use_bias": self.use_bias,
                "dropout": self.dropout_rate,
                "kernel_initializer": self.kernel_initializer,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
