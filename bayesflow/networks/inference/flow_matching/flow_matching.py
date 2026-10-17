from typing import Any
from collections.abc import Sequence

import keras

from bayesflow.distributions import Distribution
from bayesflow.types import Shape, Tensor
from bayesflow.utils import (
    expand_right_as,
    find_network,
    integrate,
    integrate_stochastic,
    jacobian_trace,
    logging,
    optimal_transport,
    resolve_seed,
    weighted_mean,
)
from bayesflow.utils.jacobian.jacobian_trace import hutchinson_projectors
from bayesflow.utils.serialization import serialize, serializable
from bayesflow.masking import InferenceMasks

from ...inference import InferenceNetwork
from ...defaults import (
    TIME_MLP_DEFAULTS,
    DIFFUSION_TRANSFORMER_DEFAULTS,
    FLOW_MATCHING_INTEGRATE_DEFAULTS,
    OPTIMAL_TRANSPORT_DEFAULTS,
)


@serializable("bayesflow.networks")
class FlowMatching(InferenceNetwork):
    """Optimal-transport flow matching for simulation-based inference.

    Implements Optimal Transport Flow Matching, originally introduced as Rectified
    Flow, with ideas incorporated from [1-5].

    The model learns a velocity field that transports samples from a base
    distribution to the target posterior. It supports optional mini-batch optimal
    transport via the Sinkhorn algorithm [6-8] for improved training stability.

    Parameters
    ----------
    subnet : str, type, or keras.Layer
        A neural network type for the flow matching model, will be instantiated
        using *subnet_kwargs*.  If a string is provided, it should be a registered
        name (e.g., ``"time_mlp"``).  If a type or ``keras.Layer`` is provided, it
        will be directly instantiated with the given *subnet_kwargs*.  Any subnet
        must accept a tuple of tensors ``(target, time, conditions)``.
    base_distribution : str or Distribution
        The base probability distribution from which samples are drawn.
        Default is ``"normal"``.
    use_optimal_transport : bool
        Whether to apply optimal transport for improved training stability.
        Default is ``False``.  Note: this will increase training time by
        approximately 2.5×, but may lead to faster inference.
    loss_fn : str or keras.Loss
        The loss function used for training.  Default is ``"mse"``.
    integrate_kwargs : dict[str, Any], optional
        Additional keyword arguments for the ODE integrator used at inference time.
    optimal_transport_kwargs : dict[str, Any], optional
        Additional keyword arguments for configuring optimal transport.
    subnet_kwargs : dict[str, Any], optional
        Keyword arguments passed to the subnet constructor or used to update the
        default MLP settings.
    time_power_law_alpha : float
        Changes the distribution of sampled times during training.  Time is sampled
        from a power-law distribution ``p(t) ~ t^(1/(1+alpha))``, where
        ``alpha`` is the provided value.  Default is 0.5.
    fixed_target_prob : float
        Probability of fixing each target during training (so the network learns arbitrary
        conditionals). Default is 0.0.
    missing_target_prob : float
        Probability of marking each fixed target as missing during training, so the network
        learns to handle marginalization of targets. Only takes effect when the subnet
        accepts ``infer_target_mask`` (e.g. ``diffusion_transformer``). Default is 0.0.
    missing_conditions_prob : float
        Probability of marking each condition as missing during training, so the network learns
        to handle missing conditions. Only takes effect when the subnet accepts
        ``observed_condition_mask`` (e.g. ``diffusion_transformer``). Default is 0.0.
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``.

    References
    ----------
    [1] Liu et al. (2022). Flow straight and fast: Learning to generate and
        transfer data with rectified flow. arXiv:2209.03003.
    [2] Lipman et al. (2022). Flow matching for generative modeling.
        arXiv:2210.02747.
    [3] Tong et al. (2023). Improving and generalizing flow-based generative
        models with minibatch optimal transport. arXiv:2302.00482.
    [4] Wildberger et al. (2023). Flow matching for scalable simulation-based
        inference. NeurIPS, 36, 16837-16864.
    [5] Orsini et al. (2025). Flow matching posterior estimation for
        simulation-based atmospheric retrieval of exoplanets. IEEE Access.
    [6] Nguyen et al. (2022). Improving Mini-batch Optimal Transport via Partial
        Transportation.
    [7] Cheng et al. (2025). The Curse of Conditions: Analyzing and Improving
        Optimal Transport for Conditional Flow-Based Generation.
    [8] Fluri et al. (2024). Improving Flow Matching for Simulation-Based
        Inference.
    """

    def __init__(
        self,
        subnet: str | type | keras.Layer = "time_mlp",
        base_distribution: str | Distribution = "normal",
        use_optimal_transport: bool = False,
        loss_fn: str | keras.Loss = "mse",
        integrate_kwargs: dict[str, Any] = None,
        optimal_transport_kwargs: dict[str, Any] = None,
        subnet_kwargs: dict[str, Any] = None,
        time_power_law_alpha: float = 0.5,
        fixed_target_prob: float = 0.0,
        missing_target_prob: float = 0.0,
        missing_conditions_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            base_distribution,
            fixed_target_prob=fixed_target_prob,
            missing_target_prob=missing_target_prob,
            missing_conditions_prob=missing_conditions_prob,
            **kwargs,
        )

        self.use_optimal_transport = use_optimal_transport

        self.integrate_kwargs = FLOW_MATCHING_INTEGRATE_DEFAULTS | (integrate_kwargs or {})
        self.optimal_transport_kwargs = OPTIMAL_TRANSPORT_DEFAULTS | (optimal_transport_kwargs or {})

        self.loss_fn = keras.losses.get(loss_fn)

        self.time_power_law_alpha = time_power_law_alpha
        if self.time_power_law_alpha <= -1.0:
            raise ValueError("'time_power_law_alpha' must be greater than -1.0.")

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "time_mlp":
            subnet_kwargs = TIME_MLP_DEFAULTS | subnet_kwargs
        if subnet == "diffusion_transformer":
            subnet_kwargs = DIFFUSION_TRANSFORMER_DEFAULTS | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self.output_projector = None

    def compute_metrics(
        self,
        x: Tensor | Sequence[Tensor],
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        if isinstance(x, Sequence):
            x0, x1, t, x, target_velocity = x
        else:
            x1 = x
            x0 = self.base_distribution.sample(keras.ops.shape(x1)[:-1], seed=self.seed_generator)

            if self.use_optimal_transport:
                provided_masks = self.mask_handler.collect_mask_kwargs(kwargs)
                if provided_masks:
                    names = ", ".join(provided_masks)
                    raise ValueError(
                        "Explicit per-example masks cannot be combined with optimal transport because "
                        f"the transport assignment would reorder them ({names}). Use generated masks or "
                        "disable optimal transport."
                    )
                # we must choose between resampling x0 or x1
                # since the data is possibly noisy and may contain outliers, it is better
                # to possibly drop some samples from x1 than from x0
                # in the marginal over multiple batches, this is not a problem
                x0, x1, conditions = optimal_transport(
                    x0,
                    x1,
                    conditions=conditions,
                    seed=self.seed_generator,
                    **self.optimal_transport_kwargs,
                )

            u = keras.random.uniform((keras.ops.shape(x0)[0],), seed=self.seed_generator)
            # p(t) ∝ t^(1/(1+α)), the inverse CDF: F^(-1)(u) = u^(1+α), α=0 is uniform
            t = u ** (1 + self.time_power_law_alpha)
            t = expand_right_as(t, x0)

            x = t * x1 + (1 - t) * x0
            target_velocity = x1 - x0

        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            stage=stage,
            subnet=self.subnet,
        )
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, x1)

        predicted_velocity = self._velocity(
            x,
            time=t,
            conditions=conditions,
            training=stage == "training",
            masks=masks,
        )

        loss = self.loss_fn(
            self.mask_handler.maybe_apply(target_velocity, masks.fixed_target_mask),
            self.mask_handler.maybe_apply(predicted_velocity, masks.fixed_target_mask),
        )
        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)

        # construct input shape for subnet and subnet projector
        time_shape = (xz_shape[0], 1)  # same batch dims, 1 feature
        self.subnet.build((xz_shape, time_shape, conditions_shape))
        out_shape = self.subnet.compute_output_shape((xz_shape, time_shape, conditions_shape))
        if out_shape[-1] != xz_shape[-1]:
            self.output_projector = keras.layers.Dense(
                units=xz_shape[-1],
                bias_initializer="zeros",
                name="output_projector",
            )
        else:
            self.output_projector = keras.layers.Identity()
        self.output_projector.build(out_shape)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "subnet": self.subnet,
            "base_distribution": self.base_distribution,
            "use_optimal_transport": self.use_optimal_transport,
            "loss_fn": self.loss_fn,
            "integrate_kwargs": self.integrate_kwargs,
            "optimal_transport_kwargs": self.optimal_transport_kwargs,
            "time_power_law_alpha": self.time_power_law_alpha,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)

    def velocity(
        self, xz: Tensor, time: float | Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor:
        masks = self.mask_handler.resolve(
            kwargs,
            targets=xz,
            conditions=conditions,
            subnet=self.subnet,
        )
        return self._velocity(
            xz,
            time=time,
            conditions=conditions,
            training=training,
            masks=masks,
        )

    def _velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
    ) -> Tensor:
        time = keras.ops.convert_to_tensor(time, dtype=keras.ops.dtype(xz))
        time = expand_right_as(time, xz)
        time = keras.ops.broadcast_to(time, keras.ops.shape(xz)[:-1] + (1,))
        subnet_out = self.subnet((xz, time, conditions), training=training, **masks.subnet_kwargs)
        out = self.output_projector(subnet_out)

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            out = self.mask_handler.maybe_apply(out, masks.fixed_target_mask)
        return out

    def _velocity_trace(
        self,
        xz: Tensor,
        time: Tensor,
        conditions: Tensor = None,
        projectors: Tensor = None,
        training: bool = False,
        masks: InferenceMasks = None,
    ) -> tuple[Tensor, Tensor]:
        def f(x):
            return self._velocity(
                x,
                time=time,
                conditions=conditions,
                training=training,
                masks=masks,
            )

        v, trace = jacobian_trace(f, xz, projectors=projectors, return_output=True)

        return v, keras.ops.expand_dims(trace, axis=-1)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            subnet=self.subnet,
        )

        runtime_kwargs = self.mask_handler.without_mask_kwargs(kwargs)
        seed = resolve_seed(runtime_kwargs.pop("seed", None), self.seed_generator)
        hutchinson_steps = runtime_kwargs.pop("hutchinson_steps", None)

        # Build integrate kwargs: instance config -> call-time overrides
        integrate_kwargs = self.integrate_kwargs | runtime_kwargs

        if density and integrate_kwargs["method"] == "glass":
            logging.warning("GLASS is not supported for density computation. Falling back to tsit5 ODE solver.")
            integrate_kwargs["method"] = "tsit5"

        # During a forward pass the observed sample supplies fixed values unless
        # the caller provided an explicit value (for example after adapting and
        # standardizing a partially observed record).
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)

        if density:
            # one probe per solve, fixed over time, keeps the augmented ODE deterministic
            projectors = hutchinson_projectors(x, hutchinson_steps, seed=seed)

            def deltas(time, xz):
                v, trace = self._velocity_trace(
                    xz,
                    time=time,
                    conditions=conditions,
                    projectors=projectors,
                    training=training,
                    masks=masks,
                )
                return {"xz": v, "trace": trace}

            state = {"xz": x, "trace": keras.ops.zeros(keras.ops.shape(x)[:-1] + (1,), dtype=keras.ops.dtype(x))}
            state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **integrate_kwargs)

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) + keras.ops.squeeze(
                state["trace"], axis=-1
            )

            return z, log_density

        def deltas(time, xz):
            return {
                "xz": self._velocity(
                    xz,
                    time=time,
                    conditions=conditions,
                    training=training,
                    masks=masks,
                )
            }

        state = {"xz": x}
        if integrate_kwargs["method"] == "glass":
            project_state, project_updates = self.mask_handler.integrator_callbacks(masks, replacement=x)

            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=None,
                state=state,
                start_time=1.0,
                stop_time=0.0,
                noise_schedule="flow_matching",
                seed=seed,
                state_projection_fn=project_state,
                state_update_projection_fn=project_updates,
                **integrate_kwargs,
            )
        else:
            state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **integrate_kwargs)

        z = state["xz"]

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        masks = self.mask_handler.resolve(
            kwargs,
            targets=z,
            conditions=conditions,
            subnet=self.subnet,
        )

        runtime_kwargs = self.mask_handler.without_mask_kwargs(kwargs)
        seed = resolve_seed(runtime_kwargs.pop("seed", None), self.seed_generator)
        hutchinson_steps = runtime_kwargs.pop("hutchinson_steps", None)

        # Build integrate kwargs: instance config -> call-time overrides
        integrate_kwargs = self.integrate_kwargs | runtime_kwargs

        if density and integrate_kwargs["method"] == "glass":
            logging.warning("GLASS is not supported for density computation. Falling back to tsit5 ODE solver.")
            integrate_kwargs["method"] = "tsit5"

        z = self.mask_handler.maybe_apply(z, masks.fixed_target_mask, masks.fixed_target_value)

        if density:
            # one probe per solve, fixed over time, keeps the augmented ODE deterministic
            projectors = hutchinson_projectors(z, hutchinson_steps, seed=seed)

            def deltas(time, xz):
                v, trace = self._velocity_trace(
                    xz,
                    time=time,
                    conditions=conditions,
                    projectors=projectors,
                    training=training,
                    masks=masks,
                )
                return {"xz": v, "trace": trace}

            state = {"xz": z, "trace": keras.ops.zeros(keras.ops.shape(z)[:-1] + (1,), dtype=keras.ops.dtype(z))}
            state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **integrate_kwargs)

            x = state["xz"]
            log_density = self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) - keras.ops.squeeze(
                state["trace"], axis=-1
            )

            return x, log_density

        def deltas(time, xz):
            return {
                "xz": self._velocity(
                    xz,
                    time=time,
                    conditions=conditions,
                    training=training,
                    masks=masks,
                )
            }

        state = {"xz": z}
        if integrate_kwargs["method"] == "glass":
            project_state, project_updates = self.mask_handler.integrator_callbacks(masks)
            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=None,
                state=state,
                start_time=0.0,
                stop_time=1.0,
                seed=seed,
                noise_schedule="flow_matching",
                state_projection_fn=project_state,
                state_update_projection_fn=project_updates,
                **integrate_kwargs,
            )
        else:
            state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **integrate_kwargs)

        x = state["xz"]

        return x
