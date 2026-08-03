from collections.abc import Mapping, Sequence
from typing import Literal, Tuple

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, TimeSeriesTransformer, TransformerDecoder
from bayesflow.types import Tensor
from bayesflow.utils import call_accepts_kwarg, split_arrays
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serialize, serializable

from .continuous_approximator import ContinuousApproximator
from .helpers import AutoregressiveConditionBuilder, AutoregressiveSampler


@serializable("bayesflow.approximators")
class AutoregressiveApproximator(ContinuousApproximator):
    r"""Estimate a joint smoothing distribution with an ordinary inference network.

    The approximator factorizes

    .. math::

        p(\theta_{1:T}\mid x_{1:T}) = \prod_{t=1}^T
        p(\theta_t\mid x_{1:T}, \theta_{1:t-1}).

    An encoder represents the complete observation sequence. A causal decoder
    combines that representation with shifted targets to produce one condition
    per time point. Training and density evaluation operate on complete sequence
    tensors; sampling advances the decoder autoregressively with attention caches.
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

        return_decoder_time = call_accepts_kwarg(self.decoder_network.initialize_cache, "time")
        resolved_conditions = self.condition_builder.resolve(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            decoder_network=self.decoder_network,
            inference_variables=None,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            summary_attention_mask=adapted.get("summary_attention_mask"),
            summary_mask=adapted.get("summary_mask"),
            return_decoder_time=return_decoder_time,
        )
        if return_decoder_time:
            _, encoder_outputs, decoder_time = resolved_conditions
        else:
            _, encoder_outputs = resolved_conditions
            decoder_time = None

        kwargs = self._maybe_standardize_fixed_target_value(kwargs)
        kwargs = self._maybe_inject_guidance_unstandardize(kwargs)
        samples = self.sampler.sample(
            inference_network=self.inference_network,
            decoder_network=self.decoder_network,
            num_samples=num_samples,
            conditions=encoder_outputs,
            batch_size=batch_size,
            sample_shape=sample_shape,
            seed=resolve_seed(seed),
            num_steps=keras.ops.shape(encoder_outputs)[1],
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
        log_det = keras.ops.convert_to_tensor(adapter_log_det.get("inference_variables", 0.0))
        step_log_prob = step_log_prob + log_det + standardizer_log_det
        if adapted.get("inference_mask") is not None:
            step_log_prob = step_log_prob * keras.ops.cast(
                adapted["inference_mask"],
                step_log_prob.dtype,
            )
        return keras.ops.convert_to_numpy(keras.ops.sum(step_log_prob, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.pop("summary_network", None)
        return config | serialize(
            {
                "encoder_network": self.encoder_network,
                "decoder_network": self.decoder_network,
            }
        )
