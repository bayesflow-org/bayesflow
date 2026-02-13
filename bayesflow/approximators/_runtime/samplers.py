from typing import Sequence, Literal

from tqdm.auto import tqdm

import keras

from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils.logging import warning
from bayesflow.utils import slice_maybe_nested, dim_maybe_nested, tree_concatenate
from bayesflow.types import Tensor


@serializable("bayesflow.approximators")
class Sampler:
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
            return None, None

        shape = keras.ops.shape(conditions)
        batch_size = shape[0]
        non_batch_dims = shape[1:]

        conditions = keras.ops.expand_dims(conditions, axis=1)
        conditions = keras.ops.broadcast_to(conditions, (batch_size, num_samples, *non_batch_dims))
        conditions = keras.ops.reshape(conditions, (batch_size * num_samples, *non_batch_dims))

        return conditions

    def unflatten_samples(self, samples: Tensor, num_samples: int):
        return keras.ops.reshape(samples, (-1, num_samples, *keras.ops.shape(samples)[1:]))

    def sample(
        self,
        inference_network: keras.Layer,
        num_samples: int,
        conditions: Tensor | None = None,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Sequence[int] | int = "infer",
        **kwargs,
    ):
        if conditions is None:
            return self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=None,
                sample_shape=sample_shape,
                **kwargs,
            )

        num_conditions = dim_maybe_nested(conditions, axis=0)

        if batch_size is None:
            batch_size = num_conditions

        batches = []
        for i in tqdm(range(0, num_conditions, batch_size), desc="Sampling", unit="batch"):
            batch_conditions = slice_maybe_nested(conditions, i, i + batch_size)

            batch_samples = self._sample_batch(
                inference_network=inference_network,
                num_samples=num_samples,
                conditions=batch_conditions,
                sample_shape=sample_shape,
                **kwargs,
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
        **kwargs,
    ):
        conditions = self.repeat_and_flatten_conditions(conditions, num_samples)

        if conditions is None:
            batch_shape = (num_samples,)
        else:
            # conditions already flattened to (batch_size*num_samples, ...)
            batch_shape = (keras.ops.shape(conditions)[0],)

        sample_shape = self.infer_sample_shape(conditions, sample_shape)
        batch_shape = batch_shape + sample_shape

        samples = inference_network.sample(batch_shape, conditions=conditions, **kwargs)

        if conditions is not None:
            samples = self.unflatten_samples(samples, num_samples)
        return samples

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "Sampler":
        return cls(**deserialize(config, custom_objects=custom_objects))
