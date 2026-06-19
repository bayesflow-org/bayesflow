import keras

from functools import lru_cache

from bayesflow._backend import jacrev
from bayesflow.networks.inference import InferenceNetwork
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, issue_url
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..autoencoder import AutoEncoder


@serializable("bayesflow.experimental")
class LatentIN(InferenceNetwork):
    def __init__(self, autoencoder: AutoEncoder, inference_network: InferenceNetwork, **kwargs):
        super().__init__(**kwargs)

        if isinstance(inference_network, LatentIN):
            raise TypeError("Cannot recurse latent inference networks.")

        self.autoencoder = autoencoder
        self.inference_network = inference_network

    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            return

        shape = xz_shape
        self.autoencoder.build(shape)
        shape = self.autoencoder.compute_output_shape(shape)
        self.inference_network.build(shape, conditions_shape)

        self.base_distribution.build(shape)

        self.built = True

    def get_config(self):
        base_config = super().get_config()
        config = {
            "autoencoder": self.autoencoder,
            "inference_network": self.inference_network,
        }

        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def _forward(self, x, conditions=None, density=False, training=False, **kwargs):
        if not density:
            y = self.autoencoder(x, training=training, **kwargs)
            z = self.inference_network(y, conditions=conditions, training=training, **kwargs)
            return z

        @lru_cache(1)
        def fn(_x):
            return self.autoencoder(_x, inverse=False, training=training, **filter_kwargs(kwargs, self.autoencoder))

        # the general case is that the encoder projects down, so we use jacrev to save on compute as compared to jacfwd
        jac_fn = jacrev(fn, argnums=0, has_aux=False)

        y = fn(x)
        jac = jac_fn(x)

        z, log_density = self.inference_network(y, inverse=False, density=True, training=training, **kwargs)

        # modified change of variables; p(x) = p(f(x)) / sqrt(|J^T J|)
        log_density = log_density - 0.5 * keras.ops.logdet(jac.T @ jac)

        return z, log_density

    def _inverse(self, z, conditions=None, density=False, training=False, **kwargs):
        if not density:
            y = self.inference_network(z, conditions=conditions, inverse=True, training=training, **kwargs)
            x = self.autoencoder(y, training=training, inverse=True, **kwargs)
            return x

        raise NotImplementedError(
            f"Inverse mode density estimation is not supported; use forward-mode estimation instead. "
            f"If you need this feature, please open an issue at {issue_url}."
        )

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        autoencoder_metrics = self.autoencoder.compute_metrics(x, sample_weight=sample_weight, stage=stage, **kwargs)

        try:
            y = autoencoder_metrics.pop("z")
        except KeyError:
            raise RuntimeError(
                f"The autoencoder for {self.__class__.__name__} must return a cached latent vector from its "
                f"`compute_metrics` method under the key 'z'."
            )

        if not keras.ops.is_tensor(y):
            raise TypeError(
                f"Expected the autoencoder's cached latent vector (key 'z') to be a Tensor, got {type(y)!r} instead."
            )

        # decouple the autoencoder and inference network to avoid representation drift issues
        y = keras.ops.stop_gradient(y)

        inference_metrics = self.inference_network.compute_metrics(
            y, conditions=conditions, sample_weight=sample_weight, stage=stage, **kwargs
        )

        metrics = {"loss": inference_metrics["loss"] + autoencoder_metrics["loss"]}
        autoencoder_metrics = {
            f"{self.autoencoder.__class__.__name__}/{key}": value for key, value in autoencoder_metrics.items()
        }
        inference_metrics = {
            f"{self.inference_network.__class__.__name__}/{key}": value for key, value in inference_metrics.items()
        }

        metrics = metrics | autoencoder_metrics | inference_metrics

        return metrics
