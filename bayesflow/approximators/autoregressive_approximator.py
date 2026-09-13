from collections.abc import Mapping, Sequence
from typing import Literal, Tuple

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, TimeSeriesTransformer
from bayesflow.networks.decoders import RecurrentDecoder
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
        Use `keras.layers.Identity()` for already encoded `inference_conditions`.
        Global conditions of shape `(batch, features)` become one memory token.
    decoder_network : keras.Layer or None, optional
        Causal network used to combine encoded conditions with shifted targets.
        If `None`, a `RecurrentDecoder` is used. The default transformer encoder
        and recurrent decoder require one encoded condition per target position.
    standardize : str, sequence of str, or None, optional
        Variables to standardize. Defaults to `"inference_variables"`.
    eos_value : float, sequence of float, or None, optional
        Enable learned end-of-sequence decisions with a reserved packet value.
        In adapted `inference_variables` (before dynamic standardization), an
        entire packet row matching this scalar or vector denotes EOS. A scalar
        broadcasts across features; a vector can be e.g. `[-999, 0, 0]`.
        NaN markers are supported. An explicit boolean
        `inference_mask` takes precedence: true rows are packets, the first false
        row is EOS, and the remaining false rows are padding. Masks must be
        contiguous prefixes and training/evaluation sequences need a terminal
        slot. If an adapter transforms parameters, supply the mask before those
        transforms or specify the sentinel in the adapter's output space.
        EOS densities require a shared packetwise parameter bijection.
        The default is None, preserving fixed-length behavior.

        In EOS mode decoder positions are packet indices, not observation times.
        The default recurrent decoder requires aligned memory, including the
        terminal slot. Repeat global conditions across the maximum horizon, or
        explicitly use `TransformerDecoder(include_condition=False)` for
        independent observation and packet lengths.
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
        eos_value: float | Sequence[float] | None = None,
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

        self.eos_value = eos_value
        self.eos_head = keras.layers.Dense(1) if eos_value is not None else None
        self.decoder_network = decoder_network if decoder_network is not None else RecurrentDecoder()
        self.condition_builder = AutoregressiveConditionBuilder()
        self.sampler = AutoregressiveSampler()

    def build(self, data_shapes: Mapping[str, tuple]):
        self._build_standardization_layers(data_shapes)

        inference_shape = tuple(data_shapes["inference_variables"])
        if len(inference_shape) != 3 or inference_shape[1] == 0:
            raise ValueError("inference_variables must have shape (batch, sequence_length, packet_features).")
        summary_shape = data_shapes.get("summary_variables")
        encoder_input_shape = self.condition_builder.encoder_input_shape(
            summary_shape,
            data_shapes.get("inference_conditions"),
        )

        if not self.encoder_network.built:
            self.encoder_network.build(encoder_input_shape)
        encoder_output_shape = self.encoder_network.compute_output_shape(encoder_input_shape)
        self._validate_decoder_horizon(inference_shape[1], encoder_output_shape[1])

        if not self.decoder_network.built:
            self.decoder_network.build(inference_shape, encoder_output_shape)
        decoder_output_shape = self.decoder_network.compute_output_shape(inference_shape, encoder_output_shape)
        if self.eos_head is not None and not self.eos_head.built:
            self.eos_head.build(decoder_output_shape)

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
        inference_variables, inference_mask, eos_mask, complete = self._prepare_sequence(
            inference_variables, inference_mask
        )
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
            use_encoder_time=self.eos_head is None,
            summary_attention_mask=summary_attention_mask,
            summary_mask=summary_mask,
            inference_attention_mask=inference_attention_mask,
            inference_mask=inference_mask,
        )

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables,
            conditions=conditions,
            sample_weight=self._sequence_weights(inference_mask, sample_weight, inference_variables.dtype),
            stage=stage,
        )

        loss = inference_metrics.pop("loss")
        eos_metrics = {}
        if self.eos_head is not None:
            eos_nll = -self._eos_step_log_prob(conditions, inference_mask)
            eos_weights = self._sequence_weights(eos_mask, sample_weight, eos_nll.dtype)
            eos_loss = keras.ops.mean(keras.ops.sum(eos_nll * eos_weights, axis=1))
            # Inference networks average over the rectangular batch/step axes.
            # Restore a sequence sum so both terms have the joint NLL scale.
            loss = loss * keras.ops.cast(keras.ops.shape(inference_variables)[1], loss.dtype) + eos_loss
            loss = keras.ops.where(keras.ops.all(complete), loss, keras.ops.cast(float("inf"), loss.dtype))
            eos_metrics = {"eos_loss": eos_loss}

        inference_metrics = {
            f"{self.inference_network.__class__.__name__}/{key}": value for key, value in inference_metrics.items()
        }

        return self._with_layer_losses(loss) | inference_metrics | eos_metrics

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        max_horizon: int | None = None,
        return_summaries: bool = False,
        seed: int | keras.random.SeedGenerator | None = None,
        to_numpy: bool = True,
        **kwargs,
    ) -> dict[str, np.ndarray | Tensor]:
        """Sample packets in parallel across draws, sequentially across positions.

        In EOS mode supply either `sample_shape` or `max_horizon`: a positive
        cap on decoding positions, not a known packet count. `max_horizon` also
        caps an explicit `sample_shape` when both are supplied. Without EOS,
        the default `sample_shape="infer"` infers length from encoder memory.
        Returns the inverse-adapted
        parameter arrays plus `_lengths` of shape `(batch, num_samples)`, `_mask`
        of shape `(batch, num_samples, horizon)`, and `_truncated` of shape
        `(batch, num_samples)`. A true truncation flag means no EOS was drawn
        before the cap. EOS/padding is filled with `eos_value` in adapted space
        before applying the inverse adapter; metadata is not transformed.

        Important: Do not pass a known `inference_mask` when sampling learned lengths.
        """
        if sample_shape == "infer":
            if max_horizon is not None:
                sample_shape = max_horizon
            elif self.eos_head is not None:
                raise ValueError("EOS sampling requires an explicit maximum horizon via sample_shape or max_horizon.")

        adapted = self.adapter(conditions, strict=False, stage="inference")

        encoder_outputs, decoder_time = self.condition_builder.resolve_encoder(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            use_encoder_time=self.eos_head is None,
            summary_attention_mask=adapted.get("summary_attention_mask"),
            summary_mask=adapted.get("summary_mask"),
        )

        kwargs = self._maybe_standardize_fixed_target_value(kwargs)
        kwargs = self._maybe_inject_guidance_unstandardize(kwargs)
        sample_shape = self.sampler.infer_sample_shape(encoder_outputs, sample_shape)
        if len(sample_shape) == 1:
            if max_horizon is not None:
                sample_shape = (min(sample_shape[0], max_horizon),)
            self._validate_decoder_horizon(sample_shape[0], encoder_outputs.shape[1], sampling=True)

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
            eos_head=self.eos_head,
            **kwargs,
        )

        metadata = {}
        if self.eos_head is not None:
            metadata = {key: value for key, value in samples.items() if key != "values"}
            samples = samples["values"]

        samples = keras.tree.map_structure(
            lambda value: self.standardizer.maybe_standardize(
                value,
                key="inference_variables",
                stage="inference",
                forward=False,
            ),
            samples,
        )
        if self.eos_head is not None:
            samples = keras.ops.where(
                metadata["_mask"][..., None], samples, keras.ops.convert_to_tensor(self.eos_value, dtype=samples.dtype)
            )
        samples = keras.tree.map_structure(
            lambda value: self.adapter(
                {"inference_variables": value},
                inverse=True,
                strict=False,
            ),
            samples,
        )

        if split:
            samples = split_arrays(samples, axis=-1)
            if not to_numpy:
                samples = keras.tree.map_structure(keras.ops.convert_to_tensor, samples)
        samples.update(metadata)
        if return_summaries:
            samples["_summaries"] = encoder_outputs
        if to_numpy:
            samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)
        return samples

    def log_prob(
        self,
        data: Mapping[str, np.ndarray],
        to_numpy: bool = True,
        **kwargs,
    ) -> np.ndarray | Tensor:
        if self.eos_head is None:
            adapted, adapter_log_det = self.adapter(data, strict=False, log_det_jac=True, stage="inference")
        else:
            adapted = self.adapter(data, strict=False, stage="inference")
            adapter_log_det = {}
        raw_variables, inference_mask, eos_mask, complete = self._prepare_sequence(
            adapted.get("inference_variables"), adapted.get("inference_mask")
        )
        inference_variables, standardizer_log_det = self.standardizer.maybe_standardize(
            raw_variables,
            key="inference_variables",
            stage="inference",
            log_det_jac=True,
            mask=inference_mask,
        )

        conditions, _ = self.condition_builder.resolve(
            standardizer=self.standardizer,
            encoder_network=self.encoder_network,
            decoder_network=self.decoder_network,
            inference_variables=inference_variables,
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            stage="inference",
            use_encoder_time=self.eos_head is None,
            summary_attention_mask=adapted.get("summary_attention_mask"),
            summary_mask=adapted.get("summary_mask"),
            inference_attention_mask=adapted.get("inference_attention_mask"),
            inference_mask=inference_mask,
        )

        inference_kwargs = {key: value for key, value in kwargs.items() if key != "batch_size"}
        step_log_prob = self.inference_network.log_prob(
            inference_variables,
            conditions=conditions,
            **inference_kwargs,
        )
        step_log_prob = step_log_prob + standardizer_log_det
        adapter_log_det = keras.ops.cast(
            adapter_log_det.get("inference_variables", 0.0),
            step_log_prob.dtype,
        )
        if self.eos_head is not None and self.adapter.transforms:
            # Use the existing adapter API with each packet as a batch item.
            # A singleton time axis preserves as_time_series inverse transforms.
            packets = keras.ops.reshape(raw_variables, (-1, 1, keras.ops.shape(raw_variables)[-1]))
            raw_packets = self.adapter({"inference_variables": packets}, inverse=True, strict=False)
            _, packet_log_det = self.adapter(raw_packets, strict=False, log_det_jac=True, stage="inference")
            if "inference_variables" in packet_log_det:
                step_log_prob += keras.ops.reshape(
                    packet_log_det["inference_variables"], keras.ops.shape(step_log_prob)
                )

        if inference_mask is not None:
            step_log_prob = keras.ops.where(inference_mask, step_log_prob, keras.ops.zeros_like(step_log_prob))
        log_prob = keras.ops.sum(step_log_prob, axis=-1) + adapter_log_det

        if self.eos_head is not None:
            eos_log_prob = self._eos_step_log_prob(conditions, inference_mask)
            log_prob += keras.ops.sum(keras.ops.where(eos_mask, eos_log_prob, 0.0), axis=1)
            log_prob = keras.ops.where(complete, log_prob, keras.ops.cast(float("-inf"), log_prob.dtype))

        if to_numpy:
            log_prob = keras.ops.convert_to_numpy(log_prob)
        return log_prob

    def _prepare_sequence(self, inference_variables, inference_mask):
        """Separate packet values, terminal decisions, and ignored padding."""
        inference_variables = keras.ops.convert_to_tensor(inference_variables)
        if len(inference_variables.shape) != 3:
            raise ValueError("inference_variables must have shape (batch, sequence_length, packet_features).")

        if inference_mask is None and self.eos_head is not None:
            marker = keras.ops.convert_to_tensor(self.eos_value, dtype=inference_variables.dtype)
            matches = keras.ops.logical_or(
                inference_variables == marker,
                keras.ops.logical_and(keras.ops.isnan(inference_variables), keras.ops.isnan(marker)),
            )
            inference_mask = keras.ops.logical_not(keras.ops.all(matches, axis=-1))

        if inference_mask is None:
            return inference_variables, None, None, None
        inference_mask = keras.ops.cast(inference_mask, "bool")

        if len(inference_mask.shape) != 2:
            raise ValueError("inference_mask must have shape (batch, sequence_length).")

        for actual, expected in zip(inference_mask.shape, inference_variables.shape[:-1]):
            if actual is not None and expected is not None and actual != expected:
                raise ValueError("inference_mask must match the batch and sequence dimensions of inference_variables.")

        eos_mask, complete = None, None
        if self.eos_head is not None:
            prefix_mask = keras.ops.cumprod(keras.ops.cast(inference_mask, "int32"), axis=1) > 0
            complete = keras.ops.logical_and(
                keras.ops.all(inference_mask == prefix_mask, axis=1), keras.ops.logical_not(prefix_mask[:, -1])
            )
            inference_mask = prefix_mask
            eos_mask = keras.ops.concatenate([keras.ops.ones_like(prefix_mask[:, :1]), prefix_mask[:, :-1]], axis=1)

        # Never feed an extreme sentinel or NaN padding into the continuous model.
        inference_variables = keras.ops.where(
            inference_mask[..., None], inference_variables, keras.ops.zeros_like(inference_variables)
        )
        return inference_variables, inference_mask, eos_mask, complete

    @staticmethod
    def _sequence_weights(mask, sample_weight, dtype):
        if mask is None:
            return sample_weight
        weights = keras.ops.cast(mask, dtype)
        if sample_weight is not None:
            sample_weight = keras.ops.cast(sample_weight, dtype)
            if keras.ops.ndim(sample_weight) == 1:
                sample_weight = sample_weight[:, None]
            weights = weights * sample_weight
        return weights

    def _eos_step_log_prob(self, conditions, inference_mask):
        logits = self.eos_head(conditions)
        targets = keras.ops.cast(keras.ops.logical_not(inference_mask)[..., None], logits.dtype)
        return -keras.ops.binary_crossentropy(targets, logits, from_logits=True)[..., 0]

    def _validate_decoder_horizon(self, horizon, memory_length, sampling=False):
        aligned = isinstance(self.decoder_network, RecurrentDecoder) or getattr(
            self.decoder_network, "include_condition", False
        )
        if aligned and horizon is not None and memory_length is not None:
            invalid = horizon > memory_length if sampling else horizon != memory_length
            if invalid:
                raise ValueError(
                    "This decoder requires time-aligned conditions. Use "
                    "TransformerDecoder(include_condition=False) for independent memory and packet lengths."
                )

    def get_config(self):
        config = super().get_config()
        config.pop("summary_network", None)
        return config | serialize(
            {
                "encoder_network": self.encoder_network,
                "decoder_network": self.decoder_network,
                "eos_value": self.eos_value,
            }
        )
