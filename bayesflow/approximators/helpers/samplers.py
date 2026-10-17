from typing import Sequence, Literal

from tqdm.auto import tqdm

import keras

from bayesflow.masking import MaskHandler, MaskName
from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils.logging import warning
from bayesflow.utils import (
    dim_maybe_nested,
    filter_kwargs,
    repeat_and_flatten,
    slice_maybe_nested,
    tree_concatenate,
)
from bayesflow.types import Tensor


@serializable("bayesflow.approximators")
class Sampler:
    """Handles batched, repeated sampling from an inference network.

    Orchestrates the full sampling pipeline:

    1. Repeat and flatten conditions so each condition is paired with
       ``num_samples`` independent draws.
    2. Infer or validate the structural ``sample_shape``.
    3. Call ``inference_network.sample``.
    4. Unflatten the resulting samples back to
       ``(batch_size, num_samples, ...)``.

    Supports optional mini-batching over conditions (controlled by
    ``batch_size``) to manage memory for large sample counts.
    """

    def infer_sample_shape(
        self,
        conditions: Tensor | None,
        sample_shape: Literal["infer"] | Sequence[int] | int,
    ):
        if sample_shape == "infer":
            if conditions is None:
                warning("No conditions to infer sample_shape from. Assuming no structural dimensions.")
                return ()
            return tuple(keras.ops.shape(conditions)[1:-1])

        if isinstance(sample_shape, int):
            return (sample_shape,)

        if isinstance(sample_shape, (tuple, list)):
            return tuple(sample_shape)

        raise ValueError(
            f"sample_shape must be 'infer', an int, or a tuple/list of ints, but got {type(sample_shape)}."
        )

    def repeat_and_flatten_conditions(self, conditions: Tensor | None, num_samples: int):
        if conditions is None:
            return None

        return repeat_and_flatten(conditions, num_samples)

    def unflatten_samples(self, samples, num_samples: int):
        return keras.tree.map_structure(
            lambda s: keras.ops.reshape(s, (-1, num_samples, *keras.ops.shape(s)[1:])),
            samples,
        )

    def sample(
        self,
        inference_network: keras.Layer,
        num_samples: int,
        conditions: Tensor | None = None,
        batch_size: int | None = None,
        data_batch_size: int | None = None,
        sample_shape: Literal["infer"] | Sequence[int] | int = "infer",
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        if conditions is None:
            if data_batch_size is not None:
                sample_shape = self.infer_sample_shape(None, sample_shape)
                batch_size = data_batch_size if batch_size is None else batch_size
                batched_keys = MaskHandler.batched_keys(
                    kwargs,
                    data_batch_size,
                    target_structural_rank=len(sample_shape),
                )
                batches = []
                for i in tqdm(range(0, data_batch_size, batch_size), desc="Sampling", unit="batch"):
                    current_batch_size = min(batch_size, data_batch_size - i)
                    batch_kwargs = {
                        key: slice_maybe_nested(value, i, i + current_batch_size) if key in batched_keys else value
                        for key, value in kwargs.items()
                    }
                    batches.append(
                        self._sample_batch(
                            inference_network=inference_network,
                            num_samples=num_samples,
                            conditions=None,
                            data_batch_size=current_batch_size,
                            sample_shape=sample_shape,
                            seed=seed,
                            batched_keys=batched_keys,
                            **batch_kwargs,
                        )
                    )
                return tree_concatenate(batches, axis=0)

            return self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=None,
                sample_shape=sample_shape,
                seed=seed,
                **kwargs,
            )

        num_conditions = dim_maybe_nested(conditions, axis=0)

        if num_conditions is None:
            return self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=conditions,
                sample_shape=sample_shape,
                seed=seed,
                **kwargs,
            )

        if batch_size is None:
            batch_size = num_conditions

        target_structural_rank = len(self.infer_sample_shape(conditions, sample_shape))
        condition_structural_rank = len(self.infer_sample_shape(conditions, "infer"))

        batched_keys = MaskHandler.batched_keys(
            kwargs,
            num_conditions,
            target_structural_rank=target_structural_rank,
            condition_structural_rank=condition_structural_rank,
        )

        batches = []
        for i in tqdm(range(0, num_conditions, batch_size), desc="Sampling", unit="batch"):
            batch_conditions = slice_maybe_nested(conditions, i, i + batch_size)
            batch_kwargs = {
                key: slice_maybe_nested(value, i, i + batch_size) if key in batched_keys else value
                for key, value in kwargs.items()
            }

            batch_samples = self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=batch_conditions,
                sample_shape=sample_shape,
                seed=seed,
                batched_keys=batched_keys,
                **batch_kwargs,
            )
            batches.append(batch_samples)

        return tree_concatenate(batches, axis=0)

    def _sample_batch(
        self,
        *,
        inference_network: keras.Layer,
        num_samples: int,
        conditions: Tensor | None,
        sample_shape: Literal["infer"] | Sequence[int] | int,
        data_batch_size: int | None = None,
        batched_keys: set[str] | None = None,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        condition_batch_size = dim_maybe_nested(conditions, axis=0) if conditions is not None else data_batch_size
        target_structural_rank = len(self.infer_sample_shape(conditions, sample_shape))
        condition_structural_rank = 0 if conditions is None else len(self.infer_sample_shape(conditions, "infer"))
        conditions = self.repeat_and_flatten_conditions(conditions, num_samples)

        # tensors like fixed_target_mask (shape [feature_dim]) are passed through
        # unchanged when no conditions are given
        if batched_keys is None:
            batched_keys = MaskHandler.batched_keys(
                kwargs,
                condition_batch_size,
                target_structural_rank=target_structural_rank,
                condition_structural_rank=condition_structural_rank,
            )
        kwargs = {
            key: self.repeat_and_flatten_conditions(value, num_samples) if key in batched_keys else value
            for key, value in kwargs.items()
        }

        if conditions is None:
            batch_shape = (num_samples if data_batch_size is None else data_batch_size * num_samples,)
        else:
            # conditions already flattened to (batch_size*num_samples, ...)
            batch_shape = (keras.ops.shape(conditions)[0],)

        sample_shape = self.infer_sample_shape(conditions, sample_shape)
        batch_shape = batch_shape + sample_shape

        samples = inference_network.sample(batch_shape, conditions=conditions, seed=seed, **kwargs)

        if conditions is not None or data_batch_size is not None:
            samples = self.unflatten_samples(samples, num_samples)
        return samples

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Sampler":
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("bayesflow.approximators")
class AutoregressiveSampler(Sampler):
    """Sample trajectories with an incrementally cached autoregressive decoder."""

    def _sample_batch(
        self,
        *,
        inference_network: keras.Layer,
        decoder_network: keras.Layer,
        num_samples: int,
        conditions: Tensor | None,
        sample_shape: Literal["infer"] | Sequence[int] | int,
        data_batch_size: int | None = None,
        batched_keys: set[str] | None = None,
        time: Tensor | None = None,
        encoder_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        seed: keras.random.SeedGenerator | int | None = None,
        **kwargs,
    ):
        condition_batch_size = dim_maybe_nested(conditions, axis=0) if conditions is not None else data_batch_size
        encoder_outputs = self.repeat_and_flatten_conditions(conditions, num_samples)
        time = self.repeat_and_flatten_conditions(time, num_samples)
        encoder_mask = self.repeat_and_flatten_conditions(encoder_mask, num_samples)
        target_mask = self.repeat_and_flatten_conditions(target_mask, num_samples)
        target_attention_mask = self.repeat_and_flatten_conditions(target_attention_mask, num_samples)

        sample_shape = self.infer_sample_shape(encoder_outputs, sample_shape)
        if len(sample_shape) != 1:
            raise ValueError(
                "Autoregressive sampling requires exactly one structural sample dimension "
                f"for the sequence length, but got {sample_shape}."
            )
        num_steps = sample_shape[0]

        if batched_keys is None:
            condition_structural_rank = 0 if conditions is None else len(self.infer_sample_shape(conditions, "infer"))
            batched_keys = MaskHandler.batched_keys(
                kwargs,
                condition_batch_size,
                target_structural_rank=1,
                condition_structural_rank=condition_structural_rank,
            )
        kwargs = {
            key: self.repeat_and_flatten_conditions(value, num_samples) if key in batched_keys else value
            for key, value in kwargs.items()
        }

        cache_kwargs = filter_kwargs(
            {
                "encoder_mask": encoder_mask,
                "time": time,
            },
            decoder_network.initialize_cache,
        )
        cache_kwargs = {key: value for key, value in cache_kwargs.items() if value is not None}
        cache = decoder_network.initialize_cache(encoder_outputs, **cache_kwargs)
        decode_kwargs = filter_kwargs(
            {
                "target_mask": target_mask,
                MaskName.ATTENTION_MASK: target_attention_mask,
            },
            decoder_network.decode_step,
        )
        decode_kwargs = {key: value for key, value in decode_kwargs.items() if value is not None}
        previous_target = None
        generated = []

        for step in range(num_steps):
            step_conditions, cache = decoder_network.decode_step(
                previous_target,
                step=step,
                cache=cache,
                **decode_kwargs,
            )
            step_kwargs = {
                key: MaskHandler.select_autoregressive_step(
                    key,
                    value,
                    step=step,
                    num_steps=num_steps,
                    has_batch_axis=key in batched_keys,
                )
                if hasattr(value, "shape")
                else value
                for key, value in kwargs.items()
            }

            flat_batch_size = (
                keras.ops.shape(encoder_outputs)[0] if encoder_outputs is not None else data_batch_size * num_samples
            )
            current_target = inference_network.sample(
                (flat_batch_size,),
                conditions=step_conditions,
                seed=seed,
                **step_kwargs,
            )

            if target_mask is not None:
                current_target = current_target * keras.ops.cast(
                    target_mask[:, step : step + 1],
                    current_target.dtype,
                )

            generated.append(current_target)
            previous_target = current_target

        return self.unflatten_samples(keras.ops.stack(generated, axis=1), num_samples)
