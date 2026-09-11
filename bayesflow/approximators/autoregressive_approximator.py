from collections.abc import Mapping, Sequence
from typing import Literal, Tuple

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, TimeSeriesTransformer
from bayesflow.networks.decoders import TransformerDecoder
from bayesflow.types import Tensor
from bayesflow.utils import split_arrays
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serialize, serializable

from .continuous_approximator import ContinuousApproximator
from .helpers import AutoregressiveConditionBuilder, AutoregressiveSampler


@serializable("bayesflow.approximators")
class AutoregressiveApproximator(ContinuousApproximator):
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
        Additional keyword arguments passed to `ContinuousApproximator`.
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

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables,
            conditions=conditions,
            sample_weight=sample_weight,
            stage=stage,
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
        **kwargs,
    ) -> dict[str, np.ndarray]:
        if conditions is None:
            raise ValueError("Autoregressive sampling requires summary_variables.")

        adapted = self.adapter(conditions, strict=False, stage="inference")
        adapted = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted)

        encoder_outputs, decoder_time = self.condition_builder.resolve_encoder(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            summary_attention_mask=adapted.get("summary_attention_mask"),
            summary_mask=adapted.get("summary_mask"),
        )

        kwargs = self._maybe_standardize_fixed_target_value(kwargs)
        kwargs = self._maybe_inject_guidance_unstandardize(kwargs)

        samples = self.sampler.sample(
            inference_network=self.inference_network,
            decoder_network=self.decoder_network,
            num_samples=num_samples,
            conditions=encoder_outputs,
            batch_size=batch_size,
            sample_shape=sample_shape,
            seed=resolve_seed(seed, self.seed_generator),
            time=decoder_time,
            encoder_mask=adapted.get("summary_mask"),
            target_mask=adapted.get("inference_mask"),
            target_attention_mask=adapted.get("inference_attention_mask"),
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
                {"inference_variables": keras.ops.convert_to_numpy(value)},
                inverse=True,
                strict=False,
            ),
            samples,
        )

        if return_summaries:
            samples["_summaries"] = keras.ops.convert_to_numpy(encoder_outputs)
        if split:
            samples = split_arrays(samples, axis=-1)
        return samples

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        adapted, adapter_log_det = self.adapter(
            data,
            strict=False,
            log_det_jac=True,
            stage="inference",
        )
        adapted = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted)
        inference_variables, standardizer_log_det = self.standardizer.maybe_standardize(
            adapted.get("inference_variables"),
            key="inference_variables",
            stage="inference",
            log_det_jac=True,
            mask=adapted.get("inference_mask"),
        )

        conditions, _ = self.condition_builder.resolve(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            decoder_network=self.decoder_network,
            inference_variables=inference_variables,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            summary_attention_mask=adapted.get("summary_attention_mask"),
            summary_mask=adapted.get("summary_mask"),
            inference_attention_mask=adapted.get("inference_attention_mask"),
            inference_mask=adapted.get("inference_mask"),
        )

        inference_kwargs = {key: value for key, value in kwargs.items() if key != "batch_size"}
        step_log_prob = self.inference_network.log_prob(
            inference_variables,
            conditions=conditions,
            **inference_kwargs,
        )
        step_log_prob = step_log_prob + standardizer_log_det
        if adapted.get("inference_mask") is not None:
            step_log_prob = step_log_prob * keras.ops.cast(
                adapted["inference_mask"],
                step_log_prob.dtype,
            )
        log_prob = keras.ops.sum(step_log_prob, axis=-1)
        adapter_log_det = keras.ops.cast(
            keras.ops.convert_to_tensor(adapter_log_det.get("inference_variables", 0.0)),
            log_prob.dtype,
        )
        return keras.ops.convert_to_numpy(log_prob + adapter_log_det)

    def get_config(self):
        config = super().get_config()
        config.pop("summary_network", None)
        return config | serialize(
            {
                "encoder_network": self.encoder_network,
                "decoder_network": self.decoder_network,
            }
        )
