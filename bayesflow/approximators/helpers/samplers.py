from typing import Sequence, Literal

from tqdm.auto import tqdm

import keras

from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils.logging import warning
from bayesflow.utils import (
    MaskName,
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
        sample_shape: Literal["infer"] | Sequence[int] | int = "infer",
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        if conditions is None:
            return self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=None,
                sample_shape=sample_shape,
                seed=seed,
                masking_names=(
                    MaskName.FIXED_TARGET,
                    MaskName.FIXED_TARGET_VALUE,
                    MaskName.INFER_TARGET,
                ),  # only needed for unconditional sampling
                **kwargs,
            )

        num_conditions = dim_maybe_nested(conditions, axis=0)

        if batch_size is None:
            batch_size = num_conditions

        batches = []
        for i in tqdm(range(0, num_conditions, batch_size), desc="Sampling", unit="batch"):
            batch_conditions = slice_maybe_nested(conditions, i, i + batch_size)
            batch_kwargs = {
                k: slice_maybe_nested(v, i, i + batch_size) if hasattr(v, "shape") else v for k, v in kwargs.items()
            }

            batch_samples = self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=batch_conditions,
                sample_shape=sample_shape,
                seed=seed,
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
        masking_names: Sequence[str] = (),
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        conditions = self.repeat_and_flatten_conditions(conditions, num_samples)

        # tensors like fixed_target_mask (shape [feature_dim]) are passed through
        # unchanged when no conditions are given
        kwargs = {
            k: self.repeat_and_flatten_conditions(v, num_samples)
            if hasattr(v, "shape") and k not in masking_names
            else v
            for k, v in kwargs.items()
        }

        if conditions is None:
            batch_shape = (num_samples,)
        else:
            # conditions already flattened to (batch_size*num_samples, ...)
            batch_shape = (keras.ops.shape(conditions)[0],)

        sample_shape = self.infer_sample_shape(conditions, sample_shape)
        batch_shape = batch_shape + sample_shape

        samples = inference_network.sample(batch_shape, conditions=conditions, seed=seed, **kwargs)

        if conditions is not None:
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
        time: Tensor | None = None,
        encoder_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        target_attention_mask: Tensor | None = None,
        masking_names: Sequence[str] = (),
        seed: keras.random.SeedGenerator | int | None = None,
        **kwargs,
    ):
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

        kwargs = {
            key: self.repeat_and_flatten_conditions(value, num_samples)
            if hasattr(value, "shape") and key not in masking_names
            else value
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
                "attention_mask": target_attention_mask,
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
                key: value[:, step]
                if hasattr(value, "shape") and len(value.shape) >= 3 and value.shape[1] == num_steps
                else value
                for key, value in kwargs.items()
            }

            current_target = inference_network.sample(
                (keras.ops.shape(encoder_outputs)[0],),
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
