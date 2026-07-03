from collections.abc import Sequence

import keras
from keras import layers

from bayesflow.types import Tensor
from bayesflow.utils import feature_mask, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.helpers import FFN, FourierEmbedding


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaLN-Zero affine modulation: ``x * (1 + scale) + shift``."""
    return x * (1.0 + scale) + shift


def ada_ln_bias(width: int, residual_gate_init: float, dtype: str) -> Tensor:
    """Bias vector for adaLN modulation with small nonzero residual gates."""
    zeros = keras.ops.zeros((width,), dtype=dtype)
    gates = keras.ops.full((width,), residual_gate_init, dtype=dtype)
    return keras.ops.concatenate([zeros, zeros, gates, zeros, zeros, gates], axis=0)


def conditioning_attention_mask(
    target_inference_mask: Tensor | None,
    condition_mask: Tensor | None,
    num_targets: int,
    num_conditions: int,
    batch_size: Tensor,
    target_condition_mask: Tensor | None = None,
) -> Tensor:
    """Build the directed dependency mask for target queries over ``[targets, conditions]`` keys.

    Target tokens fall into three states: *latent* (noised targets being inferred),
    *observed* (clean present targets), and *absent* (missing targets). Condition tokens
    are *observed* when present and *absent* when missing; they are frozen context and
    never issue queries, so the mask only has target query rows. It encodes:

    * latent queries attend to every present key (observed + latent targets + present conditions),
    * observed queries attend only to observed keys (a closed, noise-independent set),
    * absent keys are never attended to, and absent queries keep only self-attention.

    Parameters
    ----------
    target_inference_mask : Tensor or None
        Per-target ``(batch, num_targets)`` with ``1`` = noised/latent, ``0`` = clean.
        ``None`` treats every target as latent.
    condition_mask : Tensor or None
        Per-condition ``(batch, num_conditions)`` with ``1`` = present, ``0`` = missing.
        ``None`` treats every condition as present.
    num_targets, num_conditions : int
        Number of target and condition tokens.
    batch_size : Tensor
        Batch size.
    target_condition_mask : Tensor or None
        Per-target ``(batch, num_targets)`` with ``1`` = present, ``0`` = missing.

    Returns
    -------
    Tensor
        A boolean ``(batch, num_targets, num_targets + num_conditions)`` mask,
        ``True`` where a target query may attend to a key.
    """
    target_latent = (
        keras.ops.ones((batch_size, num_targets), dtype="bool")
        if target_inference_mask is None
        else keras.ops.cast(target_inference_mask, "bool")
    )
    target_present = (
        keras.ops.ones((batch_size, num_targets), dtype="bool")
        if target_condition_mask is None
        else keras.ops.cast(target_condition_mask, "bool")
    )
    target_observed = keras.ops.logical_and(target_present, keras.ops.logical_not(target_latent))

    if num_conditions > 0:
        condition_present = (
            keras.ops.ones((batch_size, num_conditions), dtype="bool")
            if condition_mask is None
            else keras.ops.cast(condition_mask, "bool")
        )
        keys_observed = keras.ops.concatenate([target_observed, condition_present], axis=1)
        keys_present = keras.ops.concatenate([target_present, condition_present], axis=1)
    else:
        keys_observed = target_observed
        keys_present = target_present

    o_i = target_observed[:, :, None]
    p_i = target_present[:, :, None]
    o_j = keys_observed[:, None, :]
    p_j = keys_present[:, None, :]
    eye = keras.ops.cast(keras.ops.eye(num_targets), "bool")
    if num_conditions > 0:
        eye = keras.ops.concatenate([eye, keras.ops.zeros((num_targets, num_conditions), dtype="bool")], axis=1)
    eye = eye[None]

    # Present queries: observed -> observed keys; latent -> all present keys.
    present_block = keras.ops.logical_or(
        keras.ops.logical_and(o_i, o_j),
        keras.ops.logical_and(keras.ops.logical_not(o_i), p_j),
    )
    # Absent queries keep only self-attention to avoid fully-masked rows.
    return keras.ops.where(p_i, present_block, eye)


class QKNormMultiHeadAttention(layers.MultiHeadAttention):
    """Multi-head attention with RMS-normalized queries and keys (QK-norm)."""

    def build(self, query_shape, value_shape, key_shape=None):
        super().build(query_shape, value_shape, key_shape)
        head_shape = (None, None, self._num_heads, self._key_dim)
        self.query_norm = keras.layers.RMSNormalization(axis=-1)
        self.query_norm.build(head_shape)
        self.key_norm = keras.layers.RMSNormalization(axis=-1)
        self.key_norm.build(head_shape)

    def _compute_attention(self, query, key, value, *args, **kwargs):
        return super()._compute_attention(self.query_norm(query), self.key_norm(key), value, *args, **kwargs)


@serializable("bayesflow.networks")
class TimeTransformerBlock(keras.Layer):
    """Transformer block with time conditioning.

    Target tokens are the residual stream; optional condition tokens enter only as
    frozen keys/values (cross-attention style).

    Parameters
    ----------
    width : int
        Token embedding width.
    num_heads : int, optional
        Number of attention heads. Default is ``4``.
    dropout : float, optional
        Dropout rate used in attention and feedforward sublayers. Default is
        ``0.0``.
    expansion_factor : float, optional
        Feedforward expansion factor. Default is ``4.0``.
    glu_variant : str, optional
        Gated activation variant for the feedforward network. One of
        ``"swiglu"``, ``"geglu"``, ``"reglu"``, or ``"liglu"``. Default is
        ``"swiglu"``.
    use_bias : bool, optional
        Whether dense projections include a bias term. Default is ``False``.
    residual_gate_init : float, optional
        Initial value for the adaLN residual gates. Default is ``1e-2``.
    kernel_initializer : str or keras.Initializer, optional
        Initializer for dense projection kernels. Default is
        ``"glorot_uniform"``.
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.

    """

    def __init__(
        self,
        width: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.0,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        use_bias: bool = False,
        kernel_initializer: str | keras.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if width % num_heads != 0:
            raise ValueError("TimeTransformerBlock requires width to be divisible by num_heads.")

        self.width = width
        self.num_heads = num_heads
        self.dropout = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.residual_gate_init = kwargs.get("residual_gate_init", 1e-2)

        # adaLN supplies the affine part, so the norms are non-affine (DiT-style).
        self.attn_norm = keras.layers.LayerNormalization(center=False, scale=False)
        self.ffn_norm = keras.layers.LayerNormalization(center=False, scale=False)
        # MHA-internal dropout is 0, so regularize the attention output instead of the probs is faster.
        self.attn = QKNormMultiHeadAttention(
            key_dim=width // num_heads,
            num_heads=num_heads,
            dropout=0.0,
            use_bias=use_bias,
            output_shape=width,
            kernel_initializer=kernel_initializer,
        )
        self.attn_dropout = keras.layers.Dropout(dropout)
        # adaLN-single: the (shift, scale, gate) modulation for sub-layers
        self.ada_ln_table = None
        self.ffn = FFN(
            embed_dim=width,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            dropout=dropout,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )

    def build(self, input_shape):
        if self.built:
            return

        x_shape, _ = input_shape
        self.attn_norm.build(x_shape)
        self.ffn_norm.build(x_shape)
        self.attn.build(query_shape=x_shape, value_shape=x_shape)
        self.ada_ln_table = self.add_weight(shape=(6 * self.width,), initializer="zeros", name="ada_ln_table")
        if self.residual_gate_init != 0.0:
            self.ada_ln_table.assign(ada_ln_bias(self.width, self.residual_gate_init, self.ada_ln_table.dtype))
        self.ffn.build(x_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def call(
        self,
        inputs: tuple[Tensor, Tensor],
        *,
        conditions: Tensor | None = None,
        attention_mask: Tensor | None = None,
        update_mask: Tensor | None = None,
        training: bool | None = None,
    ) -> Tensor:
        x, base_mod = inputs
        residual = x

        # adaLN-single: shared modulation from the network-level MLP plus this block's offset.
        mod = base_mod + self.ada_ln_table[None]
        mod = keras.ops.expand_dims(mod, axis=1)  # broadcast over the token axis
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = keras.ops.split(mod, 6, axis=-1)

        attn_in = modulate(self.attn_norm(x, training=training), shift_attn, scale_attn)
        kv = attn_in if conditions is None else keras.ops.concatenate([attn_in, conditions], axis=1)
        attn_out = self.attn(attn_in, kv, attention_mask=attention_mask, training=training)
        h = x + gate_attn * self.attn_dropout(attn_out, training=training)

        ffn_in = modulate(self.ffn_norm(h, training=training), shift_ffn, scale_ffn)
        h = h + gate_ffn * self.ffn(ffn_in, training=training)

        if update_mask is not None:
            h = update_mask * h + (1.0 - update_mask) * residual

        return h

    def get_config(self):
        base_config = layer_kwargs(super().get_config())
        return base_config | serialize(
            {
                "width": self.width,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "use_bias": self.use_bias,
                "residual_gate_init": self.residual_gate_init,
                "kernel_initializer": self.kernel_initializer,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("bayesflow.networks")
class TimeTransformer(keras.Layer):
    """Time-conditioned transformer with AdaLN conditioning for the time. Needs longer training than the TimeMLP,
    but allows for masking to learn arbitrary conditionals and marginals.

    Processes three inputs: a state variable ``x``, a scalar or vector-valued
    time ``t``, and an optional conditioning variable ``conditions``.  The input
    and conditions are projected into a shared feature space matching ``TimeMLP``.

    Every entry of ``x`` and every condition dimension is tokenized per-dimension. Target
    tokens form the residual stream; condition tokens are embedded once and enter every
    block only as frozen keys/values (cross-attention style), so per-block compute scales
    with the number of targets. Three masks govern the structure:
    ``target_inference_mask`` (``1`` = noised/inferred, ``0`` = clean),
    ``target_condition_mask`` (``1`` = present, ``0`` = missing) and
    ``condition_mask`` (``1`` = present, ``0`` = missing).
    From them a directed dependency mask is built so observed inputs (present clean targets + present conditions)
    form a set that inferred targets attend to, while missing targets and conditions are excluded.
    An explicit ``attention_mask`` of shape ``(batch, num_targets, num_targets +num_conditions)`` may be
    supplied via kwargs to override the derived one.

    Parameters
    ----------
    widths : Sequence[int], optional
        Hidden widths for the transformer blocks. All widths must be equal.
        Default is ``(128, 128, 128)``.
    time_embedding_dim : int, optional
        Dimensionality of the learned time embedding. Default is ``32``.
        Set to ``1`` to use time directly without embedding.
    time_emb : keras.Layer or None, optional
        Custom time embedding layer. If ``None``, uses random Fourier features.
    fourier_scale : float, optional
        Frequency scaling for the default Fourier embedding. Default is ``30.0``.
        Ignored when *time_emb* is provided.
    num_heads : int, optional
        Number of attention heads. Default is ``4``.
    dropout : float, optional
        Dropout rate used in transformer blocks. Default is ``0.05``.
    expansion_factor : float, optional
        Feedforward expansion factor. Default is ``4.0``.
    glu_variant : str, optional
        Gated activation variant for the feedforward network in each transformer
        block. One of ``"swiglu"``, ``"geglu"``, ``"reglu"``, or ``"liglu"``.
        Default is ``"swiglu"``.
    use_bias : bool, optional
        Whether dense projections include a bias term. Default is ``False``.
    kernel_initializer : str or keras.Initializer, optional
        Initializer for dense projection kernels. Default is
        ``"glorot_uniform"``.
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.
    """

    # Condition-state embedding indices for the learnable state table.
    _STATE_LATENT = 0  # target token being inferred (noised)
    _STATE_OBSERVED = 1  # clean target conditioned upon, or present condition
    _STATE_MISSING = 2  # missing target or condition
    _NUM_STATES = 3

    def __init__(
        self,
        widths: Sequence[int] = (128, 128, 128),
        *,
        time_embedding_dim: int = 32,
        time_emb: keras.Layer | None = None,
        fourier_scale: float = 30.0,
        num_heads: int = 4,
        dropout: float = 0.05,
        expansion_factor: float = 4.0,
        glu_variant: str = "swiglu",
        use_bias: bool = False,
        kernel_initializer: str | keras.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if len(widths) == 0:
            raise ValueError("TimeTransformer requires at least one hidden width.")
        if len(set(widths)) != 1:
            raise ValueError("TimeTransformer currently requires all widths to be equal.")
        if int(widths[0]) % num_heads != 0:
            raise ValueError("TimeTransformer requires each width to be divisible by num_heads.")

        self.widths = tuple(widths)
        self.width = int(widths[0])
        self.time_embedding_dim = time_embedding_dim
        self.time_emb = time_emb
        self.fourier_scale = fourier_scale
        self.num_heads = num_heads
        self.dropout = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.use_bias = use_bias
        self.residual_gate_init = kwargs.get("residual_gate_init", 1e-2)
        self.kernel_initializer = kernel_initializer

        if self.time_emb is None:
            if self.time_embedding_dim == 1:
                self.time_emb = keras.layers.Identity()
            else:
                self.time_emb = FourierEmbedding(
                    embed_dim=self.time_embedding_dim,
                    scale=self.fourier_scale,
                    include_identity=True,
                )
        # DiT-style shared time MLP on top of the Fourier features
        self.time_mlp_in = keras.layers.Dense(self.width, kernel_initializer=kernel_initializer)
        self.time_mlp_out = keras.layers.Dense(self.width, kernel_initializer=kernel_initializer)
        self.ada_ln_shared = keras.layers.Dense(6 * self.width, kernel_initializer="zeros", bias_initializer="zeros")

        self.value_proj = keras.layers.Dense(self.width, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.condition_proj = None
        self.condition_norm = keras.layers.RMSNormalization(axis=-1)
        # Learnable node-identifier and condition-state embeddings
        self.emb_initializer = keras.initializers.RandomNormal(stddev=0.02)
        self.target_id = None
        self.condition_id = None
        self.state_embeddings = None
        self.blocks = [
            TimeTransformerBlock(
                width=self.width,
                num_heads=num_heads,
                dropout=dropout,
                expansion_factor=expansion_factor,
                glu_variant=glu_variant,
                use_bias=use_bias,
                residual_gate_init=self.residual_gate_init,
                kernel_initializer=kernel_initializer,
            )
            for _ in self.widths
        ]
        # DiT-style final layer: non-affine norm with time-conditioned shift/scale,
        # zero-initialized so the network starts by predicting exactly zero.
        self.out_norm = keras.layers.LayerNormalization(center=False, scale=False)
        self.final_ada_ln = keras.layers.Dense(2 * self.width, kernel_initializer="zeros", bias_initializer="zeros")
        self.token_out = keras.layers.Dense(1, kernel_initializer="zeros")

    def build(self, input_shape):
        if self.built:
            return

        x_shape, t_shape, conditions_shape = input_shape
        num_targets = x_shape[-1]
        if num_targets is None:
            raise ValueError("TimeTransformer requires a known feature dimension for x.")

        token_shape = tuple(x_shape) + (1,)
        self.value_proj.build(token_shape)
        h_shape = self.value_proj.compute_output_shape(token_shape)

        t_shape = (t_shape[0], 1)
        self.time_emb.build(t_shape)
        t_fourier_shape = self.time_emb.compute_output_shape(t_shape)
        self.time_mlp_in.build(t_fourier_shape)
        t_emb_shape = self.time_mlp_in.compute_output_shape(t_fourier_shape)
        self.time_mlp_out.build(t_emb_shape)
        self.ada_ln_shared.build(t_emb_shape)

        # Per-node identifier embeddings
        self.target_id = self.add_weight(
            shape=(num_targets, self.width), initializer=self.emb_initializer, name="target_id_embedding"
        )
        # Condition-state embeddings: latent / observed / missing
        self.state_embeddings = self.add_weight(
            shape=(self._NUM_STATES, self.width), initializer=self.emb_initializer, name="state_embeddings"
        )

        if conditions_shape is not None:
            num_conditions = conditions_shape[-1]
            self.condition_proj = keras.layers.Dense(
                self.width, use_bias=self.use_bias, kernel_initializer=self.kernel_initializer
            )
            self.condition_proj.build(tuple(conditions_shape) + (1,))
            self.condition_id = self.add_weight(
                shape=(num_conditions, self.width), initializer=self.emb_initializer, name="condition_id_embedding"
            )
            self.condition_norm.build((conditions_shape[0], num_conditions, self.width))

        for block in self.blocks:
            block.build((h_shape, t_emb_shape))

        self.out_norm.build(h_shape)
        self.final_ada_ln.build(t_emb_shape)
        self.token_out.build(h_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor | None],
        training: bool | None = None,
        target_inference_mask: Tensor | None = None,
        condition_mask: Tensor | None = None,
        target_condition_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        x, t, conditions = inputs
        num_targets = x.shape[-1]
        batch_size = keras.ops.shape(x)[0]

        # Tokenize targets per-dimension; they form the residual stream.
        h = self.value_proj(keras.ops.expand_dims(x, axis=-1), training=training)
        h = h + self.target_id[None]
        h = h + self._state_embedding(target_inference_mask, target_condition_mask, num_targets, batch_size, h.dtype)

        # Conditions are embedded once and enter the blocks only as frozen keys/values.
        cond_h = None
        num_conditions = 0
        if conditions is not None and self.condition_proj is not None:
            num_conditions = conditions.shape[-1]
            cond_h = self.condition_proj(keras.ops.expand_dims(conditions, axis=-1), training=training)
            cond_h = cond_h + self.condition_id[None]
            cond_h = cond_h + self._state_embedding(
                None, condition_mask, num_conditions, batch_size, cond_h.dtype, latent=False
            )
            cond_h = self.condition_norm(cond_h, training=training)

        t = keras.ops.reshape(t, (keras.ops.shape(t)[0], -1))[:, :1]
        t_emb = self.time_mlp_out(keras.ops.silu(self.time_mlp_in(self.time_emb(t, training=training))))
        # Shared adaLN-single modulation, computed once and reused by every block.
        base_mod = self.ada_ln_shared(keras.ops.silu(t_emb), training=training)

        attention_mask = kwargs.get("attention_mask", None)
        no_mask_inputs = target_inference_mask is None and condition_mask is None and target_condition_mask is None
        if attention_mask is None and not no_mask_inputs:
            attention_mask = conditioning_attention_mask(
                target_inference_mask,
                condition_mask,
                num_targets,
                num_conditions,
                batch_size,
                target_condition_mask,
            )

        update_mask = feature_mask(target_inference_mask, x)

        for block in self.blocks:
            h = block(
                (h, base_mod),
                conditions=cond_h,
                attention_mask=attention_mask,
                update_mask=update_mask,
                training=training,
            )

        final_mod = keras.ops.expand_dims(self.final_ada_ln(keras.ops.silu(t_emb), training=training), axis=1)
        shift_out, scale_out = keras.ops.split(final_mod, 2, axis=-1)
        h = modulate(self.out_norm(h, training=training), shift_out, scale_out)
        out = self.token_out(h, training=training)
        return keras.ops.squeeze(out, axis=-1)

    def _state_embedding(
        self,
        latent_mask: Tensor | None,
        present_mask: Tensor | None,
        num_tokens: int,
        batch_size: Tensor,
        dtype: str,
        latent: bool = True,
    ) -> Tensor:
        """Per-token condition-state embedding over ``(batch, num_tokens, width)``.

        Each token is a soft blend of the learnable latent/observed/missing state vectors,
        driven by ``latent_mask`` (``1`` = noised/inferred) and ``present_mask`` (``1`` =
        present, ``0`` = missing). ``None`` masks default to all-latent / all-present. When
        ``latent`` is ``False`` (conditions), present tokens are always *observed*.
        """
        present = (
            keras.ops.ones((batch_size, num_tokens), dtype=dtype)
            if present_mask is None
            else keras.ops.cast(present_mask, dtype)
        )[..., None]
        if latent:
            latent_w = (
                keras.ops.ones((batch_size, num_tokens), dtype=dtype)
                if latent_mask is None
                else keras.ops.cast(latent_mask, dtype)
            )[..., None]
        else:
            latent_w = keras.ops.zeros((batch_size, num_tokens, 1), dtype=dtype)

        e_latent = self.state_embeddings[self._STATE_LATENT]
        e_observed = self.state_embeddings[self._STATE_OBSERVED]
        e_missing = self.state_embeddings[self._STATE_MISSING]

        present_state = latent_w * e_latent + (1.0 - latent_w) * e_observed
        return present * present_state + (1.0 - present) * e_missing

    def get_config(self):
        base_config = layer_kwargs(super().get_config())
        return base_config | serialize(
            {
                "widths": self.widths,
                "time_embedding_dim": self.time_embedding_dim,
                "time_emb": self.time_emb,
                "fourier_scale": self.fourier_scale,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "expansion_factor": self.expansion_factor,
                "glu_variant": self.glu_variant,
                "use_bias": self.use_bias,
                "residual_gate_init": self.residual_gate_init,
                "kernel_initializer": self.kernel_initializer,
            }
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config.pop("merge", None)
        return cls(**deserialize(config, custom_objects=custom_objects))
