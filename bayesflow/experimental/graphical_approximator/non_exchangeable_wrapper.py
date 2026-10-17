import warnings

import keras
from bayesflow.masking import InferenceMasks, MaskName
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Shape, Tensor
from bayesflow.utils import weighted_mean
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.experimental")  # type: ignore[missing-argument]
class NonExchangeableWrapper(InferenceNetwork):
    """Apply an inference network autoregressively over a non-exchangeable axis.

    Fixed-target and observed-condition masks are routed step-wise to the inner
    inference network. Infer-target masks are rejected because marginalized
    previous targets cannot be removed from an arbitrary summary network
    without an explicit feature-mask protocol.
    """

    def __init__(self, inference_network: InferenceNetwork, summary_network: SummaryNetwork, **kwargs):
        super().__init__(**kwargs)
        self.inference_network = inference_network
        self.summary_network = summary_network

    def _resolve_masks(
        self,
        targets: Tensor,
        conditions: Tensor | None,
        kwargs: dict,
        *,
        stage: str = "inference",
    ) -> InferenceMasks:
        provided = self.mask_handler.collect_semantic_masks(kwargs)
        if self.missing_target_prob > 0.0 or provided.get(MaskName.INFER_TARGET) is not None:
            raise NotImplementedError(
                "NonExchangeableWrapper cannot marginalize target features without an explicit "
                "feature-mask protocol on the summary network."
            )
        return self.mask_handler.resolve(
            kwargs,
            targets=targets,
            conditions=conditions,
            stage=stage,
        )

    def get_config(self):
        base_config = super().get_config()
        config = {"inference_network": self.inference_network, "summary_network": self.summary_network}

        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config):
        return cls(**deserialize(config))

    def build(self, xz_shape: Shape, conditions_shape: Shape | None = None) -> None:
        if self.built:
            return

        if not self.summary_network.built:
            self.summary_network.build(xz_shape)

        summary_output_shape = self.summary_network.compute_output_shape(xz_shape)
        self._summary_dim = summary_output_shape[-1]

        # internal xz_shape for non-exchangeable dimension is 1 internally
        if conditions_shape is None:
            conditions_shape = (*xz_shape[:-2], 1, summary_output_shape[-1])
        else:
            conditions_shape = (*xz_shape[:-2], 1, conditions_shape[-1] + summary_output_shape[-1])

        self._seq_len = xz_shape[-2]
        if not self.inference_network.built:
            self.inference_network.build((*xz_shape[:-2], 1, xz_shape[-1]), conditions_shape=conditions_shape)
        self.built = True

    def call(
        self,
        xz,
        conditions=None,
        inverse=False,
        density=False,
        training=False,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        if seed is not None:
            kwargs["seed"] = seed
        masks = self._resolve_masks(xz, conditions, kwargs)
        kwargs = self.mask_handler.without_semantic_kwargs(kwargs)
        return self._call_resolved(
            xz,
            conditions=conditions,
            inverse=inverse,
            density=density,
            training=training,
            masks=masks,
            kwargs=kwargs,
        )

    def _call_resolved(
        self,
        xz: Tensor,
        conditions: Tensor | None,
        inverse: bool,
        density: bool,
        training: bool,
        masks: InferenceMasks,
        kwargs: dict,
    ):
        if inverse:
            generated = []
            values = []
            log_densities = []
            # Sampling is inherently sequential: step i depends on the output of step i-1.
            # We accumulate actual generated samples (not the input noise) as the autoregressive context.
            for i in range(self._seq_len):
                prev = keras.ops.concatenate(generated, axis=-2) if generated else None
                conds = self._prepare_inference_conditions(xz, i, conditions, generated_samples=prev)
                step_kwargs = kwargs | self._step_mask_kwargs(masks, i, conds, conditions)
                result = self.inference_network(
                    xz[..., i : i + 1, :],
                    conditions=conds,
                    inverse=True,
                    density=density,
                    training=training,
                    **step_kwargs,
                )

                if density:
                    value_i, log_density_i = result
                    values.append(value_i)
                    log_densities.append(log_density_i)
                else:
                    values.append(result)

                generated.append(values[-1])

            if not density:
                return keras.ops.concatenate(values, axis=-2)

            return keras.ops.concatenate(values, axis=-2), keras.ops.concatenate(log_densities, axis=-2)

        all_conds = keras.ops.concatenate(
            [self._prepare_inference_conditions(xz, i, conditions) for i in range(self._seq_len)],
            axis=-2,
        )

        # Merge seq into batch for a single forward call
        n_params = xz.shape[-1]
        cond_dim = keras.ops.shape(all_conds)[-1]
        xz_flat = keras.ops.reshape(xz, (-1, 1, n_params))
        conds_flat = keras.ops.reshape(all_conds, (-1, 1, cond_dim))
        mask_kwargs = self._flat_mask_kwargs(masks, all_conds, conditions)

        result = self.inference_network(
            xz_flat,
            conditions=conds_flat,
            inverse=False,
            density=density,
            training=training,
            **(kwargs | mask_kwargs),
        )

        xz_shape = keras.ops.shape(xz)
        if not density:
            return keras.ops.reshape(result, xz_shape)

        z, log_density = result

        return keras.ops.reshape(z, xz_shape), keras.ops.reshape(log_density, xz_shape[:-1])

    @allow_batch_size
    def sample(self, batch_shape: Shape, conditions: Tensor = None, **kwargs) -> Tensor:
        seed = resolve_seed(kwargs.pop("seed", None), self.seed_generator)
        samples = self.inference_network.base_distribution.sample(batch_shape, seed=seed)
        samples = self(samples, conditions=conditions, inverse=True, density=False, seed=seed, **kwargs)

        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor | None, **kwargs) -> Tensor:
        _, log_density = self(samples, conditions=conditions, inverse=False, density=True, **kwargs)

        return log_density

    def compute_metrics(
        self,
        x: Tensor,
        conditions: Tensor | None = None,
        sample_weight: Tensor | None = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        # GraphicalApproximator invokes compute_metrics directly, bypassing
        # keras.Layer.__call__ and its automatic build step.
        if not self.built:
            conditions_shape = None if conditions is None else conditions.shape
            self.build(x.shape, conditions_shape=conditions_shape)

        masks = self._resolve_masks(
            x,
            conditions,
            kwargs,
            stage=stage,
        )
        z, log_density = self._call_resolved(
            x,
            conditions=conditions,
            inverse=False,
            density=True,
            training=stage == "training",
            masks=masks,
            kwargs=self.mask_handler.without_semantic_kwargs(kwargs),
        )
        sum_axes = tuple(range(1, log_density.ndim))
        loss = weighted_mean(keras.ops.mean(-log_density, axis=sum_axes), sample_weight)

        return {"loss": loss}

    def _target_mask_kwargs(self, masks: InferenceMasks, index: int | None = None) -> dict:
        result = {}
        for name, value in masks.semantic_kwargs().items():
            if name == MaskName.OBSERVED_CONDITION:
                continue
            if index is None:
                result[name] = keras.ops.reshape(
                    value,
                    (-1, 1, keras.ops.shape(value)[-1]),
                )
            else:
                result[name] = value[..., index : index + 1, :]
        return result

    def _condition_mask(
        self,
        observed_condition_mask: Tensor | None,
        prepared_conditions: Tensor,
        source_conditions: Tensor | None,
        index: int | None,
    ) -> Tensor | None:
        if observed_condition_mask is None or source_conditions is None:
            return None

        summary_shape = keras.ops.shape(prepared_conditions)[:-1] + (self._summary_dim,)
        summary_mask = keras.ops.ones(summary_shape, dtype=observed_condition_mask.dtype)

        if source_conditions.shape[-2] == 1:
            condition_mask = observed_condition_mask
        elif index is None:
            condition_mask = observed_condition_mask
        else:
            condition_mask = observed_condition_mask[..., index : index + 1, :]

        condition_mask = keras.ops.broadcast_to(
            condition_mask,
            keras.ops.shape(prepared_conditions)[:-1] + (keras.ops.shape(condition_mask)[-1],),
        )

        return keras.ops.concatenate([summary_mask, condition_mask], axis=-1)

    def _step_mask_kwargs(
        self,
        masks: InferenceMasks,
        index: int,
        prepared_conditions: Tensor,
        source_conditions: Tensor | None,
    ) -> dict:
        kwargs = self._target_mask_kwargs(masks, index=index)
        condition_mask = self._condition_mask(
            masks.observed_condition_mask,
            prepared_conditions,
            source_conditions,
            index,
        )
        if condition_mask is not None:
            kwargs[MaskName.OBSERVED_CONDITION] = condition_mask
        return kwargs

    def _flat_mask_kwargs(
        self,
        masks: InferenceMasks,
        prepared_conditions: Tensor,
        source_conditions: Tensor | None,
    ) -> dict:
        kwargs = self._target_mask_kwargs(masks)
        condition_mask = self._condition_mask(
            masks.observed_condition_mask,
            prepared_conditions,
            source_conditions,
            None,
        )
        if condition_mask is not None:
            kwargs[MaskName.OBSERVED_CONDITION] = keras.ops.reshape(
                condition_mask,
                (-1, 1, keras.ops.shape(condition_mask)[-1]),
            )
        return kwargs

    def _prepare_inference_conditions(
        self, xz: Tensor, i: int, conditions: Tensor | None = None, generated_samples: Tensor | None = None
    ):
        # this suppresses a warning that occurs because at index i=1, the context only has a single element,
        # which is then summarized.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You are using a softmax over axis", category=UserWarning)
            if i > 0:
                context = generated_samples if generated_samples is not None else xz[..., :i, :]
                parameter_conditions = keras.ops.expand_dims(self.summary_network(context), axis=-2)
            else:
                parameter_conditions = keras.ops.expand_dims(
                    self.summary_network(keras.ops.zeros_like(xz[..., :1, :])), axis=-2
                )

        if conditions is None:
            return parameter_conditions

        n_cond_steps = conditions.shape[-2]
        if n_cond_steps is not None and n_cond_steps == 1:
            cond_i = conditions
        else:
            cond_i = conditions[..., i : i + 1, :]

        return keras.ops.concatenate([parameter_conditions, cond_i], axis=-1)
