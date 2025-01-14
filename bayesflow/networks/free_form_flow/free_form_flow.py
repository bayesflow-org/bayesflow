import keras
from keras import ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_network, keras_kwargs, concatenate, log_jacobian_determinant, jvp, vjp

from ..inference_network import InferenceNetwork


@serializable(package="networks.free_form_flow")
class FreeFormFlow(InferenceNetwork):
    """Implements a dimensionality-preserving Free-form Flow.
    Incorporates ideas from [1-2].

    [1] Draxler, F., Sorrenson, P., Zimmermann, L., Rousselot, A., & Köthe, U. (2024).F
    ree-form flows: Make Any Architecture a Normalizing Flow.
    In International Conference on Artificial Intelligence and Statistics.

    [2] Sorrenson, P., Draxler, F., Rousselot, A., Hummerich, S., Zimmermann, L., &
    Köthe, U. (2024). Lifting Architectural Constraints of Injective Flows.
    In International Conference on Learning Representations.
    """

    def __init__(
        self,
        beta: float = 50.0,
        encoder_subnet: str | type = "mlp",
        decoder_subnet: str | type = "mlp",
        base_distribution: str = "normal",
        hutchinson_sampling: str = "qr",
        **kwargs,
    ):
        """Creates an instance of a Free-form Flow.

        Parameters:
        -----------
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
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))
        self.encoder_subnet = find_network(encoder_subnet, **kwargs.get("encoder_subnet_kwargs", {}))
        self.encoder_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")
        self.decoder_subnet = find_network(decoder_subnet, **kwargs.get("decoder_subnet_kwargs", {}))
        self.decoder_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")

        self.hutchinson_sampling = hutchinson_sampling
        self.beta = beta

        self.seed_generator = keras.random.SeedGenerator()

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)
        self.encoder_projector.units = xz_shape[-1]
        self.decoder_projector.units = xz_shape[-1]

        # construct input shape for subnet and subnet projector
        input_shape = list(xz_shape)

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.encoder_subnet.build(input_shape)
        self.decoder_subnet.build(input_shape)

        input_shape = self.encoder_subnet.compute_output_shape(input_shape)
        self.encoder_projector.build(input_shape)

        input_shape = self.decoder_subnet.compute_output_shape(input_shape)
        self.decoder_projector.build(input_shape)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if density:
            if conditions is None:
                # None cannot be batched, so supply as keyword argument
                z, log_det = log_jacobian_determinant(x, self.encode, conditions=None, training=training, **kwargs)
            else:
                # conditions should be batched, supply as positional argument
                z, log_det = log_jacobian_determinant(x, self.encode, conditions, training=training, **kwargs)

            log_density = self.base_distribution.log_prob(z) + log_det
            return z, log_density

        z = self.encode(x, conditions, training=training, **kwargs)
        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if density:
            if conditions is None:
                # None cannot be batched, so supply as keyword argument
                x, log_det = log_jacobian_determinant(z, self.decode, conditions=None, training=training, **kwargs)
            else:
                # conditions should be batched, supply as positional argument
                x, log_det = log_jacobian_determinant(z, self.decode, conditions, training=training, **kwargs)
            log_density = self.base_distribution.log_prob(z) - log_det
            return x, log_density

        x = self.decode(z, conditions, training=training, **kwargs)
        return x

    def encode(self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        if conditions is None:
            inp = x
        else:
            inp = concatenate(x, conditions, axis=-1)
        network_out = self.encoder_projector(
            self.encoder_subnet(inp, training=training, **kwargs), training=training, **kwargs
        )
        return network_out + x

    def decode(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        if conditions is None:
            inp = z
        else:
            inp = concatenate(z, conditions, axis=-1)
        network_out = self.decoder_projector(
            self.decoder_subnet(inp, training=training, **kwargs), training=training, **kwargs
        )
        return network_out + z

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

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)
        # sample random vector
        v = self._sample_v(x)

        def encode(x):
            return self.encode(x, conditions, training=stage == "training")

        def decode(z):
            return self.decode(z, conditions, training=stage == "training")

        # VJP computation
        z, vjp_fn = vjp(encode, x)
        v1 = vjp_fn(v)[0]
        # JVP computation
        x_pred, v2 = jvp(decode, (z,), (v,))

        # equivalent: surrogate = ops.matmul(ops.stop_gradient(v2[:, None]), v1[:, :, None])[:, 0, 0]
        surrogate = ops.sum((ops.stop_gradient(v2) * v1), axis=-1)
        nll = -self.base_distribution.log_prob(z)
        maximum_likelihood_loss = nll - surrogate
        reconstruction_loss = ops.sum((x - x_pred) ** 2, axis=-1)
        loss = ops.mean(maximum_likelihood_loss + self.beta * reconstruction_loss)

        return base_metrics | {"loss": loss}
