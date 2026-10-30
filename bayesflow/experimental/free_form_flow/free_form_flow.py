import keras
from keras import ops

from bayesflow.distributions import Distribution
from bayesflow.masking import InferenceMasks, MaskName
from bayesflow.networks import InferenceNetwork
from bayesflow.types import Tensor
from bayesflow.utils import (
    concatenate_valid,
    filter_kwargs,
    find_network,
    jacobian,
    jvp,
    log_abs_det,
    vjp,
    weighted_mean,
)
from bayesflow.utils.serialization import deserialize, serializable, serialize


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class FreeFormFlow(InferenceNetwork):
    """(IN) Implements a dimensionality-preserving Free-form Flow.
    Incorporates ideas from [1-2].

    Semantic target masks are resolved once per operation. Fixed coordinates
    remain identity coordinates of the flow and are excluded from its base
    density and training objective. Mask-aware subnets receive the resolved
    semantic masks through explicit ``call`` parameters.

    [1] Draxler, F., Sorrenson, P., Zimmermann, L., Rousselot, A., & Köthe, U. (2024).F
    ree-form flows: Make Any Architecture a Normalizing Flow.
    In International Conference on Artificial Intelligence and Statistics.

    [2] Sorrenson, P., Draxler, F., Rousselot, A., Hummerich, S., Zimmermann, L., &
    Köthe, U. (2024). Lifting Architectural Constraints of Injective Flows.
    In International Conference on Learning Representations.
    """

    ENCODER_MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.0,
    }

    DECODER_MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.0,
    }

    def __init__(
        self,
        beta: float = 50.0,
        encoder_subnet: str | keras.Layer = "mlp",
        decoder_subnet: str | keras.Layer = "mlp",
        base_distribution: str | Distribution = "normal",
        hutchinson_sampling: str = "qr",
        encoder_subnet_kwargs: dict = None,
        decoder_subnet_kwargs: dict = None,
        **kwargs,
    ):
        """Creates an instance of a Free-form Flow.

        Parameters
        ----------
        beta                  : float, optional, default: 50.0
        encoder_subnet        : str or type, optional, default: "mlp"
            A neural network type for the flow, will be instantiated using
            encoder_subnet_kwargs. Will be equipped with a projector to ensure
            the correct output dimension and a global skip connection.
        decoder_subnet        : str or type, optional, default: "mlp"
            A neural network type for the flow, will be instantiated using
            decoder_subnet_kwargs. Will be equipped with a projector to ensure
            the correct output dimension and a global skip connection.
        base_distribution     : str, optional, default: "normal"
            The latent distribution
        hutchinson_sampling   : str, optional, default: "qr
            One of `["sphere", "qr"]`. Select the sampling scheme for the
            vectors of the Hutchinson trace estimator.
        **kwargs              : dict, optional, default: {}
            Additional keyword arguments
        """
        super().__init__(base_distribution, **kwargs)

        encoder_subnet_kwargs = encoder_subnet_kwargs or {}
        decoder_subnet_kwargs = decoder_subnet_kwargs or {}

        if encoder_subnet == "mlp":
            encoder_subnet_kwargs = FreeFormFlow.ENCODER_MLP_DEFAULT_CONFIG.copy() | encoder_subnet_kwargs

        if decoder_subnet == "mlp":
            decoder_subnet_kwargs = FreeFormFlow.DECODER_MLP_DEFAULT_CONFIG.copy() | decoder_subnet_kwargs

        self.encoder_subnet = find_network(encoder_subnet, **encoder_subnet_kwargs)
        self.encoder_projector = None

        self.decoder_subnet = find_network(decoder_subnet, **decoder_subnet_kwargs)
        self.decoder_projector = None

        self.hutchinson_sampling = hutchinson_sampling
        self.beta = beta

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)

        self.encoder_projector = keras.layers.Dense(
            units=xz_shape[-1], bias_initializer="zeros", kernel_initializer="zeros", name="encoder_projector"
        )
        self.decoder_projector = keras.layers.Dense(
            units=xz_shape[-1], bias_initializer="zeros", kernel_initializer="zeros", name="decoder_projector"
        )

        # construct input shape for subnet and subnet projector
        input_shape = list(xz_shape)

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.encoder_subnet.build(input_shape)
        self.decoder_subnet.build(input_shape)

        output_shape = self.encoder_subnet.compute_output_shape(input_shape)
        self.encoder_projector.build(output_shape)

        output_shape = self.decoder_subnet.compute_output_shape(input_shape)
        self.decoder_projector.build(output_shape)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        kwargs.pop("seed", None)  # forward pass is deterministic; seed only used for base distribution
        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
        )
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)
        subnet_kwargs = self._subnet_kwargs(self.encoder_subnet, masks, kwargs)

        if density:
            z, jac = jacobian(
                lambda inp: self._encode(
                    inp,
                    conditions=conditions,
                    training=training,
                    masks=masks,
                    subnet_kwargs=subnet_kwargs,
                ),
                x,
                return_output=True,
            )
            log_det = log_abs_det(jac)

            log_density = self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) + log_det
            return z, log_density

        z = self._encode(
            x,
            conditions=conditions,
            training=training,
            masks=masks,
            subnet_kwargs=subnet_kwargs,
        )
        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        kwargs.pop("seed", None)  # inverse pass is deterministic; seed only used for base distribution
        masks = self.mask_handler.resolve(
            kwargs,
            targets=z,
            conditions=conditions,
        )
        subnet_kwargs = self._subnet_kwargs(self.decoder_subnet, masks, kwargs)
        z = self.mask_handler.maybe_apply(z, masks.fixed_target_mask, masks.fixed_target_value)

        if density:
            x, jac = jacobian(
                lambda inp: self._decode(
                    inp,
                    conditions=conditions,
                    training=training,
                    masks=masks,
                    subnet_kwargs=subnet_kwargs,
                ),
                z,
                return_output=True,
            )
            log_det = log_abs_det(jac)

            log_density = self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) - log_det
            return x, log_density

        x = self._decode(
            z,
            conditions=conditions,
            training=training,
            masks=masks,
            subnet_kwargs=subnet_kwargs,
        )
        return x

    def encode(self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        """Encode targets while honoring any explicitly provided semantic masks."""
        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
        )
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)
        return self._encode(
            x,
            conditions=conditions,
            training=training,
            masks=masks,
            subnet_kwargs=self._subnet_kwargs(self.encoder_subnet, masks, kwargs),
        )

    def _encode(
        self,
        x: Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
        subnet_kwargs: dict,
    ) -> Tensor:
        if conditions is None:
            inp = x
        else:
            inp = concatenate_valid([x, conditions], axis=-1)
        network_out = self.encoder_projector(self.encoder_subnet(inp, training=training, **subnet_kwargs))
        return self.mask_handler.maybe_apply(network_out + x, masks.fixed_target_mask, x)

    def decode(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        """Decode latents while honoring any explicitly provided semantic masks."""
        masks = self.mask_handler.resolve(
            kwargs,
            targets=z,
            conditions=conditions,
        )
        z = self.mask_handler.maybe_apply(z, masks.fixed_target_mask, masks.fixed_target_value)
        return self._decode(
            z,
            conditions=conditions,
            training=training,
            masks=masks,
            subnet_kwargs=self._subnet_kwargs(self.decoder_subnet, masks, kwargs),
        )

    def _decode(
        self,
        z: Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
        subnet_kwargs: dict,
    ) -> Tensor:
        if conditions is None:
            inp = z
        else:
            inp = concatenate_valid([z, conditions], axis=-1)
        network_out = self.decoder_projector(self.decoder_subnet(inp, training=training, **subnet_kwargs))
        return self.mask_handler.maybe_apply(network_out + z, masks.fixed_target_mask, z)

    def _subnet_kwargs(
        self,
        subnet: keras.Layer,
        masks: InferenceMasks,
        source: dict,
    ) -> dict:
        """Route ordinary and semantic masks only to subnets declaring them."""
        mask_kwargs = self.mask_handler.collect_subnet_masks(subnet, source, masks.semantic_kwargs())
        passthrough = {
            key: value
            for key, value in self.mask_handler.without_semantic_kwargs(source).items()
            if key not in (MaskName.ATTENTION_MASK, MaskName.MASK)
        }
        return filter_kwargs(passthrough, subnet.call) | mask_kwargs

    def _sample_v(self, x):
        batch_size = ops.shape(x)[0]
        total_dim = ops.shape(x)[-1]
        match self.hutchinson_sampling:
            case "qr":
                # Use QR decomposition as described in [2]
                v_raw = keras.random.normal((batch_size, total_dim, 1), dtype=ops.dtype(x), seed=self.seed_generator)
                q = ops.reshape(ops.qr(v_raw)[0], ops.shape(x))
                v = q * ops.sqrt(total_dim)
            case "sphere":
                # Sample from sphere with radius sqrt(total_dim), as implemented in [1]
                v_raw = keras.random.normal((batch_size, total_dim), dtype=ops.dtype(x), seed=self.seed_generator)
                v = v_raw * ops.sqrt(total_dim) / ops.sqrt(ops.sum(v_raw**2, axis=-1, keepdims=True))
            case _:
                raise ValueError(f"{self.hutchinson_sampling} is not a valid value for hutchinson_sampling.")
        return v

    def compute_metrics(
        self,
        x: Tensor,
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        training = stage == "training"
        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            stage=stage,
            subnet=self.encoder_subnet,
        )
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)
        encoder_kwargs = self._subnet_kwargs(self.encoder_subnet, masks, kwargs)
        decoder_kwargs = self._subnet_kwargs(self.decoder_subnet, masks, kwargs)

        # sample random vector
        v = self.mask_handler.maybe_apply(self._sample_v(x), masks.fixed_target_mask)

        def encode(x):
            return self._encode(
                x,
                conditions=conditions,
                training=training,
                masks=masks,
                subnet_kwargs=encoder_kwargs,
            )

        def decode(z):
            return self._decode(
                z,
                conditions=conditions,
                training=training,
                masks=masks,
                subnet_kwargs=decoder_kwargs,
            )

        # VJP computation
        z, vjp_fn = vjp(encode, x, return_output=True)
        v1 = self.mask_handler.maybe_apply(vjp_fn(v), masks.fixed_target_mask)
        # JVP computation
        x_pred, v2 = jvp(decode, (z,), (v,), return_output=True)
        v2 = self.mask_handler.maybe_apply(v2, masks.fixed_target_mask)

        # equivalent: surrogate = ops.matmul(ops.stop_gradient(v2[:, None]), v1[:, :, None])[:, 0, 0]
        surrogate = ops.sum((ops.stop_gradient(v2) * v1), axis=-1)
        nll = -self.base_distribution.log_prob(z, mask=masks.fixed_target_mask)
        maximum_likelihood_loss = nll - surrogate
        reconstruction_loss = ops.sum(
            self.mask_handler.maybe_apply((x - x_pred) ** 2, masks.fixed_target_mask),
            axis=-1,
        )

        losses = maximum_likelihood_loss + self.beta * reconstruction_loss
        loss = weighted_mean(losses, sample_weight)

        return {"loss": loss}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "beta": self.beta,
            "encoder_subnet": self.encoder_subnet,
            "decoder_subnet": self.decoder_subnet,
            "base_distribution": self.base_distribution,
            "hutchinson_sampling": self.hutchinson_sampling,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)
