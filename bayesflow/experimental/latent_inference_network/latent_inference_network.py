from functools import lru_cache

import keras

from bayesflow._backend import jacrev
from bayesflow.masking import InferenceMasks, MaskName
from bayesflow.networks.inference import InferenceNetwork
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, issue_url, log_abs_det
from bayesflow.utils.serialization import serializable, serialize
from ..autoencoder import AutoEncoder


@serializable("bayesflow.experimental")
class LatentInferenceNetwork(InferenceNetwork):
    """Inference network wrapper that operates on a compressed latent representation.

    This class wraps an autoencoder as a feature extractor, projecting input data into a lower-dimensional
    latent space before passing it to the internal inference network. This approach can significantly
    reduce computational costs, especially for high-dimensional or image data where redundancy is present.

    Condition masks are safe to delegate because conditions keep their feature
    space. Target masks are rejected: an arbitrary dimension-reducing
    autoencoder provides no well-defined map from input-feature masks to latent
    masks, so silently forwarding or clamping them would not represent the
    requested conditional density.

    The latent inference paradigm was originally introduced in [1] and is also known as "Stable Diffusion".
    To closely match this architecture, configure the network as follows:

        >>> from bayesflow.experimental import VariationalAutoEncoder
        >>> from bayesflow.networks import DiffusionModel
        >>> inference_network = LatentInferenceNetwork(VariationalAutoEncoder(...), DiffusionModel(...))

    Parameters
    ----------
    autoencoder : AutoEncoder
        The autoencoder serving as the feature extractor.
    inference_network : InferenceNetwork
        The inference network to be applied to the latent representation.

    References
    ----------
    [1]: Rombach et al. (2021). High-Resolution Image Synthesis with Latent Diffusion Models. arXiv:2112.10752.
    """

    def __init__(
        self,
        autoencoder: AutoEncoder,
        inference_network: InferenceNetwork,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(inference_network, LatentInferenceNetwork):
            raise TypeError("Cannot recurse latent inference networks.")

        self.autoencoder = autoencoder
        self.inference_network = inference_network

    def _resolve_condition_masks(
        self,
        targets: Tensor,
        conditions: Tensor | None,
        kwargs: dict,
        *,
        stage: str = "inference",
    ) -> tuple[InferenceMasks, dict]:
        """Resolve only condition masks, whose feature space is unchanged."""
        provided = self.mask_handler.collect_semantic_masks(kwargs)
        target_mask = (
            self.fixed_target_prob > 0.0
            or self.missing_target_prob > 0.0
            or any(provided.get(name) is not None for name in (MaskName.FIXED_TARGET, MaskName.INFER_TARGET))
        )
        if target_mask or provided.get(MaskName.FIXED_TARGET_VALUE) is not None:
            raise NotImplementedError(
                "LatentInferenceNetwork cannot project target masks because an arbitrary "
                "autoencoder has no well-defined mapping from input features to latent features."
            )

        masks = self.mask_handler.resolve(
            kwargs,
            targets=targets,
            conditions=conditions,
            stage=stage,
        )
        clean_kwargs = self.mask_handler.without_semantic_kwargs(kwargs)
        return masks, clean_kwargs

    @staticmethod
    def _inner_kwargs(masks: InferenceMasks, kwargs: dict) -> dict:
        return kwargs | masks.semantic_kwargs()

    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            return

        shape = xz_shape
        self.autoencoder.build(shape)
        shape = self.autoencoder.compute_output_shape(shape)

        self.inference_network.build(shape, conditions_shape)

        self.base_distribution.build(shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "autoencoder": self.autoencoder,
            "inference_network": self.inference_network,
        }

        return base_config | serialize(config)

    def _forward(
        self, x: Tensor, conditions: Tensor | None = None, density: bool = False, training: bool = False, **kwargs
    ):
        masks, kwargs = self._resolve_condition_masks(x, conditions, kwargs)
        inner_kwargs = self._inner_kwargs(masks, kwargs)

        if not density:
            y = self.autoencoder(x, training=training, **filter_kwargs(kwargs, self.autoencoder.call))
            z = self.inference_network(
                y,
                conditions=conditions,
                training=training,
                **filter_kwargs(inner_kwargs, self.inference_network.call),
            )
            return z

        @lru_cache(1)
        def fn(_x):
            return self.autoencoder(
                _x, inverse=False, training=training, **filter_kwargs(kwargs, self.autoencoder.call)
            )

        # the general case is that the encoder projects down, so we use jacrev to save on compute as compared to jacfwd
        jac_fn = jacrev(fn, argnums=0, has_aux=False)

        y = fn(x)
        jac = jac_fn(x)

        z, log_density = self.inference_network(
            y,
            inverse=False,
            density=True,
            training=training,
            **filter_kwargs(inner_kwargs, self.inference_network.call),
        )

        # modified change of variables; p(x) = p(f(x)) / sqrt(|J^T J|)
        log_density = log_density - 0.5 * log_abs_det(keras.ops.matmul(keras.ops.transpose(jac), jac))

        return z, log_density

    def _inverse(
        self, z: Tensor, conditions: Tensor | None = None, density: bool = False, training: bool = False, **kwargs
    ):
        if not self.built:
            raise RuntimeError("Must call build before calling inverse.")

        if density:
            raise NotImplementedError(
                f"Inverse mode density estimation is not supported; use forward-mode estimation instead. "
                f"If you need this feature, please open an issue at {issue_url}."
            )

        masks, kwargs = self._resolve_condition_masks(z, conditions, kwargs)
        inner_kwargs = self._inner_kwargs(masks, kwargs)

        y = self.inference_network(
            z,
            conditions=conditions,
            inverse=True,
            training=training,
            **filter_kwargs(inner_kwargs, self.inference_network.call),
        )
        x = self.autoencoder(y, training=training, inverse=True, **filter_kwargs(kwargs, self.autoencoder.call))
        return x

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        masks, kwargs = self._resolve_condition_masks(
            x,
            conditions,
            kwargs,
            stage=stage,
        )
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
            y,
            conditions=conditions,
            sample_weight=sample_weight,
            stage=stage,
            **self._inner_kwargs(masks, kwargs),
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
