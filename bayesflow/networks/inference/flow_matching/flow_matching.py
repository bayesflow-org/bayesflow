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
from bayesflow.masking import InferenceMasks, MaskHandler

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
    Flow, with ideas incorporated from [1-6].

    The model learns a velocity field that transports samples from a base
    distribution (``t = 1``) to the target posterior (``t = 0``). It supports optional
    mini-batch optimal transport via the Sinkhorn algorithm [6-8] for improved training stability.

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
        approximately 2.5x, but may lead to faster inference.
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
        Changes the distribution of sampled times during training. Time is sampled
        as ``t = 1 - u^(1+alpha)`` with ``u ~ U(0, 1)``, i.e., ``p(t) ~ (1-t)^(-alpha/(1+alpha))``.
        Negative values emphasize times near the data endpoint at ``t = 0``;
        ``alpha = 0`` is uniform. Default is -0.5.
    mask_handler : ~bayesflow.masking.MaskHandler, optional
        Handler used to generate and route masks. If ``None``, a default mask handler is created.
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
    [9] Arruda, J., Bracher, N., Köthe, U., Hasenauer, J., & Radev, S. T. (2026).
        Diffusion Models in Simulation-Based Inference: A Tutorial Review.
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
        time_power_law_alpha: float = -0.5,
        mask_handler: MaskHandler | None = None,
        **kwargs,
    ):
        super().__init__(
            base_distribution,
            mask_handler=mask_handler,
            **kwargs,
        )

        self.use_optimal_transport = use_optimal_transport

        self.integrate_kwargs = FLOW_MATCHING_INTEGRATE_DEFAULTS | (integrate_kwargs or {})
        self.optimal_transport_kwargs = OPTIMAL_TRANSPORT_DEFAULTS | (optimal_transport_kwargs or {})

        self.loss_fn = keras.losses.get(loss_fn)

        self.time_power_law_alpha = time_power_law_alpha
        if self.time_power_law_alpha <= -1.0:
            raise ValueError("'time_power_law_alpha' must be greater than -1.0.")

        subnet_defaults = {"time_mlp": TIME_MLP_DEFAULTS, "diffusion_transformer": DIFFUSION_TRANSFORMER_DEFAULTS}
        self.subnet = find_network(subnet, **(subnet_defaults.get(subnet, {}) | (subnet_kwargs or {})))

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
            x1, x0, t, x, target_velocity = x
        else:
            x0 = x
            x1 = self.base_distribution.sample(keras.ops.shape(x0)[:-1], seed=self.seed_generator)

            if self.use_optimal_transport:
                provided_masks = self.mask_handler.collect_mask_kwargs(kwargs)
                if provided_masks:
                    names = ", ".join(provided_masks)
                    raise ValueError(
                        "Explicit per-example masks cannot be combined with optimal transport because "
                        f"the transport assignment would reorder them ({names}). Use generated masks or "
                        "disable optimal transport."
                    )
                # we must choose between resampling the noise x1 or the data x0
                # since the data is possibly noisy and may contain outliers, it is better
                # to possibly drop some samples from x0 than from x1
                # in the marginal over multiple batches, this is not a problem
                x1, x0, conditions, assignments = optimal_transport(
                    x1,
                    x0,
                    conditions=conditions,
                    seed=self.seed_generator,
                    return_assignments=True,
                    **self.optimal_transport_kwargs,
                )
                if sample_weight is not None:
                    sample_weight = keras.ops.take(sample_weight, assignments, axis=0)

            u = keras.random.uniform((keras.ops.shape(x0)[0],), seed=self.seed_generator)
            # inverse CDF of F(t) = 1 - (1-t)^(1/(1+α)), i.e. p(t) ∝ (1-t)^(-α/(1+α)), α=0 is uniform
            t = 1 - u ** (1 + self.time_power_law_alpha)
            t = expand_right_as(t, x0)

            x = (1 - t) * x0 + t * x1
            target_velocity = x1 - x0

        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            stage=stage,
            subnet=self.subnet,
        )
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, x0)

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
        self.output_projector = self._build_subnet_and_projector(xz_shape, conditions_shape)

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

    def _integrate(
        self,
        xz: Tensor,
        *,
        masks: InferenceMasks,
        conditions: Tensor,
        training: bool,
        start_time: float,
        stop_time: float,
        density: bool,
        kwargs: dict,
    ) -> tuple[Tensor, Tensor | None]:
        """Integrate the velocity field from `start_time` to `stop_time`.

        Returns the final state and, if `density`, the integrated divergence of the velocity.
        """
        runtime_kwargs = self.mask_handler.without_mask_kwargs(kwargs)
        seed = resolve_seed(runtime_kwargs.pop("seed", None), self.seed_generator)
        hutchinson_steps = runtime_kwargs.pop("hutchinson_steps", None)

        # Build integrate kwargs: instance config -> call-time overrides
        integrate_kwargs = self.integrate_kwargs | runtime_kwargs | {"start_time": start_time, "stop_time": stop_time}

        if density and integrate_kwargs["method"] == "glass":
            logging.warning("GLASS is not supported for density computation. Falling back to tsit5 ODE solver.")
            integrate_kwargs["method"] = "tsit5"

        def velocity(time, xz):
            return self._velocity(xz, time=time, conditions=conditions, training=training, masks=masks)

        if density:
            # one probe per solve, fixed over time, keeps the augmented ODE deterministic
            projectors = hutchinson_projectors(xz, hutchinson_steps, seed=seed)

            def deltas(time, xz):
                def ode_velocity(x):
                    return velocity(time, x)

                v, trace = jacobian_trace(ode_velocity, xz, projectors=projectors, return_output=True)
                return {"xz": v, "trace": keras.ops.expand_dims(trace, axis=-1)}

            state = {"xz": xz, "trace": keras.ops.zeros(keras.ops.shape(xz)[:-1] + (1,), dtype=keras.ops.dtype(xz))}
            state = integrate(deltas, state, **integrate_kwargs)
            return state["xz"], keras.ops.squeeze(state["trace"], axis=-1)

        def deltas(time, xz):
            return {"xz": velocity(time, xz)}

        if integrate_kwargs["method"] == "glass":
            project_state, project_updates = self.mask_handler.integrator_callbacks(masks)
            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=None,
                state={"xz": xz},
                noise_schedule="flow_matching",
                seed=seed,
                state_projection_fn=project_state,
                state_update_projection_fn=project_updates,
                **integrate_kwargs,
            )
        else:
            state = integrate(deltas, {"xz": xz}, **integrate_kwargs)
        return state["xz"], None

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            subnet=self.subnet,
        )

        # During a forward pass the observed sample supplies fixed values unless
        # the caller provided an explicit value (for example after adapting and
        # standardizing a partially observed record).
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)

        if kwargs.get("method", self.integrate_kwargs["method"]) == "glass":
            logging.warning("GLASS only supports sampling. Falling back to tsit5 ODE solver for the forward pass.")
            kwargs = kwargs | {"method": "tsit5"}

        z, trace = self._integrate(
            x,
            masks=masks,
            conditions=conditions,
            training=training,
            start_time=0.0,
            stop_time=1.0,
            density=density,
            kwargs=kwargs,
        )
        if not density:
            return z
        return z, self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) + trace

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        masks = self.mask_handler.resolve(
            kwargs,
            targets=z,
            conditions=conditions,
            subnet=self.subnet,
        )
        z = self.mask_handler.maybe_apply(z, masks.fixed_target_mask, masks.fixed_target_value)

        x, trace = self._integrate(
            z,
            masks=masks,
            conditions=conditions,
            training=training,
            start_time=1.0,
            stop_time=0.0,
            density=density,
            kwargs=kwargs,
        )
        if not density:
            return x
        return x, self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) - trace
