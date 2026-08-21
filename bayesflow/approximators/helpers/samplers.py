from typing import Sequence, Literal

from tqdm.auto import tqdm

import keras

from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils.logging import warning
from bayesflow.utils import MaskName, slice_maybe_nested, dim_maybe_nested, repeat_and_flatten, tree_concatenate
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
