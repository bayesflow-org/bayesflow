from collections.abc import Mapping, Sequence
from typing import Literal, Tuple

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, TimeSeriesTransformer
from bayesflow.networks.decoders import TransformerDecoder
from bayesflow.masking import MaskHandler, MaskName
from bayesflow.types import Tensor
from bayesflow.utils import split_arrays
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serialize, serializable

from .basic_estimator import BasicEstimator
from .helpers import AutoregressiveConditionBuilder, AutoregressiveSampler


@serializable("bayesflow.estimators")
class AutoregressiveEstimator(BasicEstimator):
    """Estimate a joint smoothing or filtering distribution with an arbitrary inference network.

    A bidirectional encoder represents the complete conditions sequence and
    learns a smoothing representation using past and future conditions. A causal
    decoder combines this representation with shifted targets to learn the
    filtering distribution over each target given the preceding targets.
    Training and density evaluation operate on complete sequence tensors, while
    sampling advances the decoder autoregressively using efficient caching.

    Parameters
    ----------
    inference_network : InferenceNetwork
        Network used to estimate the conditional target distribution.
    adapter : Adapter or None, optional
        Adapter used to transform input data.
    encoder_network : keras.Layer or None, optional
        Network used to encode the complete conditions sequence. If `None`, a
        `TimeSeriesTransformer` with `return_sequences=True` is used.
    decoder_network : keras.Layer or None, optional
        Causal network used to combine encoded conditions with shifted targets.
        If `None`, a `TransformerDecoder` is used.
    standardize : str, sequence of str, or None, optional
        Variables to standardize. Defaults to `"inference_variables"`.
    **kwargs
        Additional keyword arguments passed to `BasicEstimator`.
    """

    def __init__(
        self,
        *,
        inference_network: InferenceNetwork,
        adapter: Adapter | None = None,
        encoder_network: keras.Layer | None = None,
        decoder_network: keras.Layer | None = None,
        standardize: str | Sequence[str] | None = "inference_variables",
        **kwargs,
    ):
        super().__init__(
            inference_network=inference_network,
            adapter=adapter,
            summary_network=None,
            standardize=standardize,
            **kwargs,
        )
        self.encoder_network = (
            encoder_network if encoder_network is not None else TimeSeriesTransformer(return_sequences=True)
        )
        self.decoder_network = decoder_network if decoder_network is not None else TransformerDecoder()
        self.condition_builder = AutoregressiveConditionBuilder()
        self.sampler = AutoregressiveSampler()

    def _prepare_step_inference_kwargs(self, kwargs: Mapping, adapted: Mapping | None = None) -> dict:
        """Prepare kwargs for the inference network behind the AR decoder.

        The adapted ``inference_mask`` and ``inference_attention_mask`` describe
        the target sequence and belong to the decoder. Canonical ``mask`` and
        ``attention_mask`` values supplied directly by the caller, however,
        describe the inference subnet and are forwarded unchanged. Semantic
        masks are discovered in adapted data and can be overridden explicitly.
        """
        adapted = adapted or {}
        kwargs = dict(kwargs)
        semantic = self.mask_handler.collect_semantic_masks(adapted, kwargs)
        semantic = self._standardize_fixed_target_value(semantic)
        if semantic.get(MaskName.INFER_TARGET) is not None or self.mask_handler.missing_target_prob > 0.0:
            raise NotImplementedError(
                "AutoregressiveEstimator requires a feature-mask-aware decoder to marginalize targets. "
                "Fixed-target conditioning is supported."
            )
        passthrough = self.mask_handler.without_semantic_kwargs(kwargs)
        passthrough.pop(MaskName.INFERENCE_ATTENTION, None)
        passthrough.pop(MaskName.INFERENCE, None)
        return passthrough | semantic

    def build(self, data_shapes: Mapping[str, tuple]):
        self._build_standardization_layers(data_shapes)

        inference_shape = tuple(data_shapes["inference_variables"])
        summary_shape = tuple(data_shapes["summary_variables"])
        encoder_input_shape = self.condition_builder.encoder_input_shape(
            summary_shape,
            data_shapes.get("inference_conditions"),
        )

        if not self.encoder_network.built:
            self.encoder_network.build(encoder_input_shape)
        encoder_output_shape = self.encoder_network.compute_output_shape(encoder_input_shape)

        if not self.decoder_network.built:
            self.decoder_network.build(inference_shape, encoder_output_shape)
        decoder_output_shape = self.decoder_network.compute_output_shape(inference_shape, encoder_output_shape)

        if not self.inference_network.built:
            self.inference_network.build(inference_shape, decoder_output_shape)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor | None = None,
        summary_variables: Tensor | None = None,
        sample_weight: Tensor | None = None,
        summary_attention_mask: Tensor | None = None,
        summary_mask: Tensor | None = None,
        inference_attention_mask: Tensor | None = None,
        inference_mask: Tensor | None = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        inference_variables = self.standardizer.maybe_standardize(
            inference_variables,
            key="inference_variables",
            stage=stage,
            mask=inference_mask,
        )

        conditions, _ = self.condition_builder.resolve(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            decoder_network=self.decoder_network,
            inference_variables=inference_variables,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            stage=stage,
            summary_attention_mask=summary_attention_mask,
            summary_mask=summary_mask,
            inference_attention_mask=inference_attention_mask,
            inference_mask=inference_mask,
        )

        inference_kwargs = self._prepare_step_inference_kwargs(MaskHandler.collect_mask_kwargs(kwargs))
        inference_metrics = self.inference_network.compute_metrics(
            inference_variables,
            conditions=conditions,
            sample_weight=sample_weight,
            stage=stage,
            **inference_kwargs,
        )

        loss = inference_metrics.pop("loss")

        inference_metrics = {
            f"{self.inference_network.__class__.__name__}/{key}": value for key, value in inference_metrics.items()
        }

        return self._with_layer_losses(loss) | inference_metrics

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray] | None = None,
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        return_summaries: bool = False,
        seed: int | keras.random.SeedGenerator | None = None,
        to_numpy: bool = True,
        **kwargs,
    ) -> dict[str, np.ndarray | Tensor]:
        if conditions is None:
            raise ValueError("Autoregressive sampling requires summary_variables.")

        adapted = self.adapter(conditions, strict=False, stage="inference")

        encoder_outputs, decoder_time = self.condition_builder.resolve_encoder(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            summary_attention_mask=adapted.get(MaskName.SUMMARY_ATTENTION),
            summary_mask=adapted.get(MaskName.SUMMARY),
        )

        kwargs = self._maybe_inject_guidance_unstandardize(kwargs)
        sequence_kwargs = MaskHandler.collect_inference_masks(adapted, kwargs)
        kwargs = self._prepare_step_inference_kwargs(kwargs, adapted=adapted)

        samples = self.sampler.sample(
            inference_network=self.inference_network,
            decoder_network=self.decoder_network,
            num_samples=num_samples,
            conditions=encoder_outputs,
            batch_size=batch_size,
            sample_shape=sample_shape,
            seed=resolve_seed(seed, self.seed_generator),
            time=decoder_time,
            encoder_mask=adapted.get(MaskName.SUMMARY),
            target_mask=sequence_kwargs.get(MaskName.MASK),
            target_attention_mask=sequence_kwargs.get(MaskName.ATTENTION_MASK),
            **kwargs,
        )

        samples = keras.tree.map_structure(
            lambda value: self.standardizer.maybe_standardize(
                value,
                key="inference_variables",
                stage="inference",
                forward=False,
            ),
            samples,
        )
        samples = keras.tree.map_structure(
            lambda value: self.adapter(
                {"inference_variables": value},
                inverse=True,
                strict=False,
            ),
            samples,
        )

        if return_summaries:
            samples["_summaries"] = encoder_outputs
        if to_numpy:
            samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)
        if split:
            samples = split_arrays(samples, axis=-1)
        return samples

    def log_prob(
        self,
        data: Mapping[str, np.ndarray],
        to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray | Tensor:
        adapted, adapter_log_det = self.adapter(
            data,
            strict=False,
            log_det_jac=True,
            stage="inference",
        )

        sequence_kwargs = MaskHandler.collect_inference_masks(adapted, kwargs)
        mask_kwargs = self.mask_handler.collect_semantic_masks(adapted, kwargs)

        inference_variables, standardizer_log_det = self.standardizer.maybe_standardize(
            adapted.get("inference_variables"),
            key="inference_variables",
            stage="inference",
            log_det_jac=True,
            mask=sequence_kwargs.get(MaskName.MASK),
            log_det_jac_mask=mask_kwargs.get(MaskName.FIXED_TARGET),
        )

        conditions, _ = self.condition_builder.resolve(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            decoder_network=self.decoder_network,
            inference_variables=inference_variables,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            summary_attention_mask=adapted.get(MaskName.SUMMARY_ATTENTION),
            summary_mask=adapted.get(MaskName.SUMMARY),
            inference_attention_mask=sequence_kwargs.get(MaskName.ATTENTION_MASK),
            inference_mask=sequence_kwargs.get(MaskName.MASK),
        )

        inference_kwargs = self._prepare_step_inference_kwargs(kwargs, adapted=adapted)
        inference_kwargs.pop("batch_size", None)
        step_log_prob = self.inference_network.log_prob(
            inference_variables,
            conditions=conditions,
            **inference_kwargs,
        )
        step_log_prob = step_log_prob + standardizer_log_det
        if sequence_kwargs.get(MaskName.MASK) is not None:
            step_log_prob = step_log_prob * keras.ops.cast(
                sequence_kwargs[MaskName.MASK],
                step_log_prob.dtype,
            )
        log_prob = keras.ops.sum(step_log_prob, axis=-1)
        adapter_log_det = keras.ops.cast(
            adapter_log_det.get("inference_variables", 0.0),
            log_prob.dtype,
        )
        log_prob = log_prob + adapter_log_det
        if to_numpy:
            log_prob = keras.ops.convert_to_numpy(log_prob)
        return log_prob

    def get_config(self):
        config = super().get_config()
        config.pop("summary_network", None)
        return config | serialize(
            {
                "encoder_network": self.encoder_network,
                "decoder_network": self.decoder_network,
            }
        )
