import keras
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs, weighted_mean
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.experimental")  # type: ignore[missing-argument]
class NonExchangeableWrapper(InferenceNetwork):
    def __init__(self, inference_network: InferenceNetwork, summary_network: SummaryNetwork, **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        self.inference_network = inference_network
        self.summary_network = summary_network

    def get_config(self):
        base_config = super().get_config()
        config = {"inference_network": self.inference_network, "summary_network": self.summary_network}

        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config):
        return cls(**deserialize(config))

    def build(self, xz_shape: Shape, conditions_shape: Shape | None = None) -> None:
        if self.built:
            return

        if not self.summary_network.built:
            raise ValueError("summary network must be built before building the wrapper.")

        summary_output_shape = self.summary_network.compute_output_shape(xz_shape)

        # internal xz_shape for non-exchangeable dimension is 1 internally
        if conditions_shape is None:
            conditions_shape = (*xz_shape[:-2], 1, summary_output_shape[-1])
        else:
            conditions_shape = (*xz_shape[:-2], 1, conditions_shape[-1] + summary_output_shape[-1])

        self._seq_len = xz_shape[-2]
        self.inference_network.build((*xz_shape[:-2], 1, xz_shape[-1]), conditions_shape=conditions_shape)
        self.built = True

    def call(self, xz, conditions=None, inverse=False, density=False, training=False, **kwargs):
        if inverse:
            # Sampling is inherently sequential: step i depends on the output of step i-1
            calls = []
            for i in range(self._seq_len):
                calls.append(
                    self.inference_network._inverse(
                        xz[..., i : i + 1, :],
                        conditions=self._prepare_inference_conditions(xz, i, conditions),
                        density=density,
                        training=training,
                        **kwargs,
                    )
                )
            if not density:
                return keras.ops.concatenate(calls, axis=-2)
            values, log_densities = zip(*calls)
            return keras.ops.concatenate(values, axis=-2), keras.ops.concatenate(log_densities, axis=-2)

        # Forward pass: all positions are independent given their conditions.
        # Precompute conditions for all positions, then make a single batched network call.
        all_conds = keras.ops.concatenate(
            [self._prepare_inference_conditions(xz, i, conditions) for i in range(self._seq_len)],
            axis=-2,
        )  # (*batch, seq, cond_dim)

        # Merge seq into batch for a single forward call
        n_params = xz.shape[-1]
        cond_dim = keras.ops.shape(all_conds)[-1]
        xz_flat = keras.ops.reshape(xz, (-1, 1, n_params))
        conds_flat = keras.ops.reshape(all_conds, (-1, 1, cond_dim))

        result = self.inference_network._forward(
            xz_flat, conditions=conds_flat, density=density, training=training, **kwargs
        )

        xz_shape = keras.ops.shape(xz)
        if not density:
            return keras.ops.reshape(result, xz_shape)
        z, log_density = result
        return keras.ops.reshape(z, xz_shape), keras.ops.reshape(log_density, xz_shape[:-1])

    @allow_batch_size
    def sample(self, batch_shape: Shape, conditions: Tensor = None, **kwargs) -> Tensor:
        samples = self.inference_network.base_distribution.sample(batch_shape)
        samples = self(samples, conditions=conditions, inverse=True, density=False, **kwargs)
        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor | None, **kwargs) -> Tensor:
        _, log_density = self(samples, conditions=conditions, inverse=False, density=True, **kwargs)
        return log_density

    def compute_metrics(
        self,
        x: Tensor,
        conditions: Tensor | None = None,
        sample_weight: Tensor | None = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        z, log_density = self(
            x,
            conditions=conditions,
            inverse=False,
            density=True,
            training=stage == "training",
        )
        # sum log-probs over sequence dimension, then average over batch
        loss = weighted_mean(keras.ops.sum(-log_density, axis=-2), sample_weight)
        return {"loss": loss}

    def _prepare_inference_conditions(self, xz: Tensor, i: int, conditions: Tensor | None = None):
        if i > 0:
            parameter_conditions = keras.ops.expand_dims(self.summary_network(xz[..., :i, :]), axis=-2)
        else:
            parameter_conditions = keras.ops.expand_dims(
                self.summary_network(keras.ops.zeros_like(xz[..., :1, :])), axis=-2
            )

        if conditions is None:
            inference_conditions = parameter_conditions
        else:
            inference_conditions = keras.ops.concatenate([parameter_conditions, conditions[..., i : i + 1, :]], axis=-1)

        return inference_conditions
