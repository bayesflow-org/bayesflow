import math
from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ...helpers import FourierEmbedding, DiffusionTransformerBlock, DiffusionTransformerReadout
from ..mlp import MLP


@serializable("bayesflow.networks")
class DiffusionTransformer(keras.Layer):
    """Time-conditioned transformer with AdaLN conditioning for the time in DiT style [1]. Needs longer training than
    the TimeMLP, but allows for masking to learn arbitrary conditionals and marginals.

    Processes three inputs: a state variable ``x``, a scalar or vector-valued
    time ``t``, and an optional conditioning variable ``conditions``.  The input
    and conditions are projected into a shared feature space matching ``TimeMLP``.

    Every entry of ``x`` and every condition dimension is tokenized per-dimension. Target
    tokens form the residual stream; condition tokens are embedded once and enter every
    block only as frozen keys/values (cross-attention style), so per-block compute scales
    with the number of targets. Three masks govern the structure:
    ``fixed_target_mask`` (``1`` = noised/inferred, ``0`` = clean),
    ``infer_target_mask`` (``1`` = present, ``0`` = missing) and
    ``observed_condition_mask`` (``1`` = present, ``0`` = missing).
    From them a directed dependency mask is built so observed inputs (present clean targets + present conditions)
    form a set that inferred targets attend to, while missing targets and conditions are excluded.
    An explicit ``attention_mask`` of shape ``(batch, num_targets, num_targets + num_conditions)`` may be
    supplied. When semantic masks are also provided, the explicit and derived attention masks are
    intersected so explicit attention cannot make missing inputs visible again.

    [1] Peebles, William, and Saining Xie. "Scalable diffusion models with transformers." 2023.

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
    residual_gate_init : float, optional
        Initial value for each block's adaLN residual gates. Default is ``1e-2``.
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
        residual_gate_init: float = 1e-2,
        kernel_initializer: str | keras.Initializer = "glorot_uniform",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if len(set(widths)) != 1:
            raise ValueError("TimeTransformer currently requires all widths to be equal.")
        if widths[0] % num_heads != 0:
            raise ValueError("TimeTransformer requires each width to be divisible by num_heads.")

        self.widths = tuple(widths)
        self.width = self.widths[0]
        self.time_embedding_dim = time_embedding_dim
        self.time_emb = time_emb
        self.fourier_scale = fourier_scale
        self.num_heads = num_heads
        self.dropout = dropout
        self.expansion_factor = expansion_factor
        self.glu_variant = glu_variant
        self.use_bias = use_bias
        self.residual_gate_init = residual_gate_init
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
        self.time_mlp = MLP(
            widths=(self.width, self.width),
            activation="silu",
            kernel_initializer=kernel_initializer,
            residual=True,
            dropout=dropout,
        )
        self.ada_shared = keras.layers.Dense(6 * self.width, kernel_initializer="zeros", bias_initializer="zeros")

        self.value_proj = keras.layers.Dense(self.width, use_bias=use_bias, kernel_initializer=kernel_initializer)
        self.condition_proj = None
        self.condition_norm = None

        # Learnable node-identifier and condition-state embeddings
        self.emb_initializer = keras.initializers.RandomNormal(stddev=0.02)
        self.target_id = None
        self.condition_id = None
        self.state_embeddings = None

        self.blocks = [
            DiffusionTransformerBlock(
                width=self.width,
                num_heads=num_heads,
                dropout=dropout,
                expansion_factor=expansion_factor,
                glu_variant=glu_variant,
                use_bias=use_bias,
                residual_gate_init=residual_gate_init,
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

        # time is flattened to (batch_size, -1) in embed
        t_shape = (t_shape[0], math.prod(t_shape[1:]))
        self.time_emb.build(t_shape)
        t_fourier_shape = self.time_emb.compute_output_shape(t_shape)
        self.time_mlp.build(t_fourier_shape)
        t_emb_shape = self.time_mlp.compute_output_shape(t_fourier_shape)
        self.ada_shared.build(t_emb_shape)

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
            self.condition_norm = keras.layers.RMSNormalization(axis=-1)
            self.condition_norm.build((conditions_shape[0], num_conditions, self.width))

        for block in self.blocks:
            block.build((h_shape, t_emb_shape))

        self.out_norm.build(h_shape)
        self.final_ada_ln.build(t_emb_shape)
        self.token_out.build(h_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def embed(
        self,
        inputs: tuple[Tensor, Tensor, Tensor | None],
        training: bool | None = None,
        attention_mask: Tensor | None = None,
        fixed_target_mask: Tensor | None = None,
        observed_condition_mask: Tensor | None = None,
        infer_target_mask: Tensor | None = None,
    ) -> tuple[Tensor, tuple]:
        """The forward pass before the first block, as ``(h, context)``.

        ``context`` holds the shared adaLN modulation, the time embedding, the condition tokens, the attention
        mask and the update mask.
        """
        x, t, conditions = inputs
        num_targets = keras.ops.shape(x)[-1]
        batch_size = keras.ops.shape(x)[0]

        if infer_target_mask is not None:
            x = keras.ops.where(
                keras.ops.cast(infer_target_mask, "bool"),
                x,
                keras.ops.zeros_like(x),
            )

        # Tokenize targets per-dimension; they form the residual stream.
        h = self.value_proj(keras.ops.expand_dims(x, axis=-1))
        h = h + self.target_id[None]
        h = h + self.state_embedding(fixed_target_mask, infer_target_mask, num_targets, batch_size)

        # Conditions are embedded once and enter the blocks only as frozen keys/values.
        cond_h = None
        num_conditions = 0
        if conditions is not None and self.condition_proj is not None:
            if observed_condition_mask is not None:
                conditions = keras.ops.where(
                    keras.ops.cast(observed_condition_mask, "bool"),
                    conditions,
                    keras.ops.zeros_like(conditions),
                )
            num_conditions = keras.ops.shape(conditions)[-1]
            cond_h = self.condition_proj(keras.ops.expand_dims(conditions, axis=-1))
            cond_h = cond_h + self.condition_id[None]
            cond_h = cond_h + self.state_embedding(
                None, observed_condition_mask, num_conditions, batch_size, latent=False
            )
            cond_h = self.condition_norm(cond_h, training=training)

        t = keras.ops.reshape(t, (keras.ops.shape(t)[0], -1))
        t_emb = self.time_mlp(self.time_emb(t), training=training)

        # Shared adaLN-single modulation, computed once and reused by every block.
        base_mod = self.ada_shared(t_emb)

        no_mask_inputs = fixed_target_mask is None and observed_condition_mask is None and infer_target_mask is None
        if not no_mask_inputs:
            semantic_attention_mask = self.conditioning_attention_mask(
                fixed_target_mask=fixed_target_mask,
                observed_condition_mask=observed_condition_mask,
                infer_target_mask=infer_target_mask,
                num_targets=num_targets,
                num_conditions=num_conditions,
                batch_size=batch_size,
            )
            if attention_mask is None:
                attention_mask = semantic_attention_mask
            else:
                attention_mask = keras.ops.logical_and(keras.ops.cast(attention_mask, "bool"), semantic_attention_mask)

        update_mask = self._feature_mask(fixed_target_mask, x)

        return h, (base_mod, t_emb, cond_h, attention_mask, update_mask)

    def compute_embed_output_shape(self, input_shape):
        """The shapes of ``(h, context)`` returned by :meth:`embed`."""
        x_shape, t_shape, conditions_shape = input_shape
        batch_size, num_targets = x_shape[0], x_shape[-1]
        num_conditions = 0 if conditions_shape is None else conditions_shape[-1]

        t_shape = (t_shape[0], math.prod(t_shape[1:]))
        t_emb_shape = self.time_mlp.compute_output_shape(self.time_emb.compute_output_shape(t_shape))
        context_shape = (
            self.ada_shared.compute_output_shape(t_emb_shape),
            t_emb_shape,
            None if conditions_shape is None else (batch_size, num_conditions, self.width),
            (batch_size, num_targets, num_targets + num_conditions),
            (batch_size, num_targets, 1),
        )
        return (batch_size, num_targets, self.width), context_shape

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor | None],
        training: bool | None = None,
        attention_mask: Tensor | None = None,
        fixed_target_mask: Tensor | None = None,
        observed_condition_mask: Tensor | None = None,
        infer_target_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        h, context = self.embed(
            inputs,
            training=training,
            attention_mask=attention_mask,
            fixed_target_mask=fixed_target_mask,
            observed_condition_mask=observed_condition_mask,
            infer_target_mask=infer_target_mask,
        )

        base_mod, _, cond_h, attention_mask, update_mask = context
        for block in self.blocks:
            h = block(
                (h, base_mod),
                conditions=cond_h,
                attention_mask=attention_mask,
                update_mask=update_mask,
                training=training,
            )

        return self.readout((h, context), training=training)

    def readout(self, inputs: tuple[Tensor, tuple], training: bool | None = None) -> Tensor:
        """The final layer, called on ``(h, context)`` after the blocks."""
        h, context = inputs
        return DiffusionTransformerReadout.final_layer(
            h, context, self.out_norm, self.final_ada_ln, self.token_out, training=training
        )

    def make_readout(self) -> DiffusionTransformerReadout:
        """A new, unbuilt final layer of the same width."""
        return DiffusionTransformerReadout(self.width)

    @staticmethod
    def _feature_mask(mask: Tensor | None, reference: Tensor) -> Tensor | None:
        """Shape a target-feature mask for transformer residual updates.

        Transformer blocks operate on a ``(batch, target, width)`` residual
        stream, so this target mask needs a trailing singleton dimension for
        broadcasting over ``width``.
        """
        if mask is None:
            return None

        mask = keras.ops.cast(mask, reference.dtype)
        if len(mask.shape) == len(reference.shape) and mask.shape[-1] == 1:
            mask = keras.ops.squeeze(mask, axis=-1)
        elif len(mask.shape) == len(reference.shape) and len(mask.shape) > 2:
            mask = keras.ops.max(mask, axis=-1)

        if len(mask.shape) == 1:
            mask = keras.ops.expand_dims(mask, axis=0)

        return keras.ops.expand_dims(mask, axis=-1)

    def state_embedding(
        self,
        latent_mask: Tensor | None,
        present_mask: Tensor | None,
        num_tokens: int,
        batch_size: Tensor,
        latent: bool = True,
    ) -> Tensor:
        """Per-token condition-state embedding over ``(batch, num_tokens, width)``.

        Each token is a soft blend of the learnable latent/observed/missing state vectors,
        driven by ``latent_mask`` (``1`` = noised/inferred) and ``present_mask`` (``1`` =
        present, ``0`` = missing). ``None`` masks default to all-latent / all-present. When
        ``latent`` is ``False`` (conditions), present tokens are always *observed*.
        """
        dtype = self.state_embeddings.dtype
        ones = keras.ops.ones((batch_size, num_tokens), dtype=dtype)
        present = (ones if present_mask is None else keras.ops.cast(present_mask, dtype))[..., None]
        latent_w = (
            (ones if latent_mask is None else keras.ops.cast(latent_mask, dtype))[..., None]
            if latent
            else keras.ops.zeros((batch_size, num_tokens, 1), dtype=dtype)
        )

        e_latent = self.state_embeddings[self._STATE_LATENT]
        e_observed = self.state_embeddings[self._STATE_OBSERVED]
        e_missing = self.state_embeddings[self._STATE_MISSING]

        present_state = latent_w * e_latent + (1.0 - latent_w) * e_observed
        return present * present_state + (1.0 - present) * e_missing

    @staticmethod
    def conditioning_attention_mask(
        fixed_target_mask: Tensor | None,
        observed_condition_mask: Tensor | None,
        infer_target_mask: Tensor | None,
        num_targets: int,
        num_conditions: int,
        batch_size: Tensor,
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
        fixed_target_mask : Tensor or None
            Per-target ``(batch, num_targets)`` with ``1`` = noised/latent, ``0`` = clean.
            ``None`` treats every target as latent.
        observed_condition_mask : Tensor or None
            Per-condition ``(batch, num_conditions)`` with ``1`` = present, ``0`` = missing.
            ``None`` treats every condition as present.
        infer_target_mask : Tensor or None
            Per-target ``(batch, num_targets)`` with ``1`` = present, ``0`` = missing.
        num_targets, num_conditions : int
            Number of target and condition tokens.
        batch_size : Tensor
            Batch size.

        Returns
        -------
        Tensor
            A boolean ``(batch, num_targets, num_targets + num_conditions)`` mask,
            ``True`` where a target query may attend to a key.
        """
        target_latent = (
            keras.ops.ones((batch_size, num_targets), dtype="bool")
            if fixed_target_mask is None
            else keras.ops.cast(fixed_target_mask, "bool")
        )
        target_present = (
            keras.ops.ones((batch_size, num_targets), dtype="bool")
            if infer_target_mask is None
            else keras.ops.cast(infer_target_mask, "bool")
        )
        target_observed = keras.ops.logical_and(target_present, keras.ops.logical_not(target_latent))

        if num_conditions > 0:
            condition_present = (
                keras.ops.ones((batch_size, num_conditions), dtype="bool")
                if observed_condition_mask is None
                else keras.ops.cast(observed_condition_mask, "bool")
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
        return cls(**deserialize(config, custom_objects=custom_objects))
