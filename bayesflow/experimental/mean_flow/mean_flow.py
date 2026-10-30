from typing import Any

import keras

from bayesflow._backend import jvp, vjp
from bayesflow.distributions import Distribution
from bayesflow.masking import InferenceMasks, MaskHandler
from bayesflow.networks import InferenceNetwork
from bayesflow.networks.defaults import (
    DIFFUSION_TRANSFORMER_DEFAULTS,
    FLOW_MATCHING_INTEGRATE_DEFAULTS,
    MEAN_FLOW_MODEL_DEFAULTS,
    OPTIMAL_TRANSPORT_DEFAULTS,
)
from bayesflow.types import Shape, Tensor
from bayesflow.utils import (
    expand_right_as,
    find_network,
    integrate,
    integrate_stochastic,
    jacobian_trace,
    log_abs_det,
    logging,
    optimal_transport,
    resolve_seed,
    weighted_mean,
)
from bayesflow.utils.jacobian.jacobian_trace import hutchinson_projectors
from bayesflow.utils.serialization import serialize, serializable


@serializable("bayesflow.experimental")
class MeanFlow(InferenceNetwork):
    """Mean flow for few-step sampling and density evaluation in simulation-based inference.

    Learns the average velocity ``u(z, s, t)`` of the flow matching path, which defines the flow map
    ``z_t = z_s + (t - s) * u(z_s, s, t)`` between any two times [1, 2], trained to be invertible [3].
    Time ``t = 0`` is the data and ``t = 1`` the base distribution. Sampling and density evaluation compose
    ``steps`` flow maps on an EDM time schedule, and the density uses the log-determinant of each map, or with
    ``density_head=True`` a learned average divergence of each map [4].

    Note: With the PyTorch backend without CUDA, ``subnet="diffusion_transformer"`` is not supported because PyTorch
    scaled-dot-product attention on CPU and MPS lacks forward-mode AD support.

    Parameters
    ----------
    subnet : str, type, or keras.Layer
        A neural network type for the mean flow model, will be instantiated
        using *subnet_kwargs*.  If a string is provided, it should be a registered
        name (e.g., ``"time_mlp"``).  If a type or ``keras.Layer`` is provided, it
        will be directly instantiated with the given *subnet_kwargs*.  Any subnet
        must accept a tuple of tensors ``(target, time, conditions)``, where ``time``
        holds the two times ``(s, s - t)``.
    base_distribution : str or Distribution
        The base probability distribution from which samples are drawn.
        Default is ``"normal"``.
    use_optimal_transport : bool
        Whether to apply optimal transport for improved training stability.
        Default is ``False``.
    loss_fn : str or keras.Loss
        The loss function of the average velocity, summed over the target dimensions, and of the invertibility
        loss. Default is ``"mse"``.
    integrate_kwargs : dict[str, Any], optional
        Additional keyword arguments for the ODE integrator used with ``use_instantaneous_flow=True``.
    optimal_transport_kwargs : dict[str, Any], optional
        Additional keyword arguments for configuring optimal transport.
    subnet_kwargs : dict[str, Any], optional
        Keyword arguments passed to the subnet constructor or used to update the
        default MLP settings.
    time_power_law_alpha : float
        Changes the distribution of sampled times during training. Time is sampled
        as ``t = u^(1+alpha)`` with ``u ~ U(0, 1)``, i.e., ``p(t) ~ t^(-alpha/(1+alpha))``.
        Positive values emphasize times near the data endpoint at ``t = 0``;
        ``alpha = 0`` is uniform. Default is 0.5.
    steps : int
        Default number of flow maps used for sampling and density evaluation. Can be overridden
        per call by passing ``steps=`` to sampling. Default is ``4``.
    velocity_head_layers : int
        Number of final subnet blocks that are duplicated into a head predicting the instantaneous
        velocity. Requires a subnet with ``embed``, ``compute_embed_output_shape`` and ``blocks``, such as
        ``"time_mlp"`` or ``"diffusion_transformer"``. With 0, the instantaneous velocity is the average
        velocity at ``s = t``. Default is 2.
    velocity_loss_fn : str or keras.Loss
        The loss function of the velocity head. Default is ``"mse"``.
    density_head : bool
        Whether to learn the average divergence of each flow map as a second output of the subnet, which then
        replaces the exact log-determinants in density evaluation. Default is ``False``.
    boundary_prob : float
        Fraction of training time pairs with ``s = t``. Default is 0.7.
    cycle_weight : float
        Weight of the invertibility loss, which maps from ``s`` to ``t`` and back. Default is 2.5.
    cycle_mismatch_prob : float
        Probability of evaluating the invertibility loss at the conditions of another sample in the batch.
        Default is 0.6.
    adaptive_loss_power : float
        Power ``p`` of the adaptive weight ``(loss + adaptive_loss_eps)^(-p)`` of the average velocity loss [2],
        0 disables the weighting. Default is 0.25.
    adaptive_loss_eps : float
        Constant of the adaptive loss weight. Default is 0.1.
    mask_handler : ~bayesflow.masking.MaskHandler, optional
        Handler used to generate and route masks. If ``None``, a default mask handler is created.
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``.

    References
    ----------
    [1] Geng et al. (2025). Mean flows for one-step generative modeling. NeurIPS, 38.
    [2] Geng et al. (2026). Improved mean flows: On the challenges of fastforward generative models. CVPR.
    [3] Rehman et al. (2026). FALCON: Few-step accurate likelihoods for continuous flows. ICLR.
    [4] OuYang et al. (2026). Few-step Boltzmann generators via scalable likelihood flow maps. arXiv:2606.29110.
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
        steps: int = 4,
        velocity_head_layers: int = 2,
        velocity_loss_fn: str | keras.Loss = "mse",
        density_head: bool = False,
        boundary_prob: float = 0.7,
        cycle_weight: float = 2.5,
        cycle_mismatch_prob: float = 0.6,
        adaptive_loss_power: float = 0.25,
        adaptive_loss_eps: float = 0.1,
        mask_handler: MaskHandler | None = None,
        **kwargs,
    ):
        super().__init__(
            base_distribution,
            mask_handler=mask_handler,
            **kwargs,
        )

        self.use_optimal_transport = use_optimal_transport
        self.loss_fn = keras.losses.get(loss_fn)
        self.velocity_loss_fn = keras.losses.get(velocity_loss_fn)

        self.integrate_kwargs = FLOW_MATCHING_INTEGRATE_DEFAULTS | (integrate_kwargs or {})
        self.optimal_transport_kwargs = OPTIMAL_TRANSPORT_DEFAULTS | (optimal_transport_kwargs or {})

        self.time_power_law_alpha = time_power_law_alpha
        if self.time_power_law_alpha <= -1.0:
            raise ValueError("'time_power_law_alpha' must be greater than -1.0.")

        self.steps = steps
        self.velocity_head_layers = velocity_head_layers
        self.density_head = density_head
        if density_head and use_optimal_transport:
            # the divergence target assumes noise drawn independently of the data
            raise ValueError("'density_head' cannot be combined with optimal transport.")

        # defaults of the call options
        self.time_schedule = None
        self.rho = 7.0
        self.sigma_min = 1e-3
        self.use_instantaneous_flow = False
        self.use_density_head = density_head
        self.boundary_prob = boundary_prob
        self.cycle_weight = cycle_weight
        self.cycle_mismatch_prob = cycle_mismatch_prob
        self.adaptive_loss_power = adaptive_loss_power
        self.adaptive_loss_eps = adaptive_loss_eps

        subnet_defaults = {
            "time_mlp": MEAN_FLOW_MODEL_DEFAULTS,
            "diffusion_transformer": DIFFUSION_TRANSFORMER_DEFAULTS
            | {"fourier_scale": MEAN_FLOW_MODEL_DEFAULTS["fourier_scale"]},
        }
        self.subnet = find_network(subnet, **(subnet_defaults.get(subnet, {}) | (subnet_kwargs or {})))
        if (density_head or velocity_head_layers > 0) and not hasattr(self.subnet, "embed"):
            raise ValueError(
                f"'density_head' and 'velocity_head_layers' require a subnet with 'embed', which "
                f"{type(self.subnet).__name__} lacks. Set 'density_head=False' and 'velocity_head_layers=0' instead."
            )

        # the velocity head shares the subnet up to the blocks it duplicates
        self.velocity_blocks = []
        if velocity_head_layers > 0:
            if velocity_head_layers > len(self.subnet.blocks):
                raise ValueError(f"'velocity_head_layers' exceeds the {len(self.subnet.blocks)} subnet blocks.")
            self.velocity_blocks = [
                type(block).from_config(block.get_config() | {"name": None})
                for block in self.subnet.blocks[-velocity_head_layers:]
            ]

        self.output_projector = None
        self.velocity_readout = None
        self.velocity_projector = None
        self.divergence_readout = None
        self.divergence_projector = None

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)

        # construct input shape for subnet and subnet projector
        time_shape = (xz_shape[0], 2)  # same batch dims, 2 features
        input_shape = (xz_shape, time_shape, conditions_shape)
        self.subnet.build(input_shape)
        self.output_projector = self._build_projector(self.subnet.compute_output_shape(input_shape), xz_shape[-1])

        if self.density_head:
            # the divergence head reads the final hidden state of the subnet
            h_shape, context_shape = self.subnet.compute_embed_output_shape(input_shape)
            for block in self.subnet.blocks:
                h_shape = block.compute_output_shape((h_shape, context_shape))
            if hasattr(self.subnet, "make_readout"):
                self.divergence_readout = self.subnet.make_readout()
                self.divergence_readout.build((h_shape, context_shape))
                self.divergence_projector = self._build_projector(
                    self.divergence_readout.compute_output_shape((h_shape, context_shape)),
                    xz_shape[-1],
                    name="divergence_projector",
                )
            else:
                # dense even at matching widths, where the identity would equal the average velocity
                self.divergence_projector = keras.layers.Dense(units=xz_shape[-1], bias_initializer="zeros")
                self.divergence_projector.build(h_shape)

        if not self.velocity_blocks:
            return

        h_shape, context_shape = self.subnet.compute_embed_output_shape(input_shape)
        for block in self.subnet.blocks[: -len(self.velocity_blocks)]:
            h_shape = block.compute_output_shape((h_shape, context_shape))
        for block in self.velocity_blocks:
            block.build((h_shape, context_shape))
            h_shape = block.compute_output_shape((h_shape, context_shape))

        if hasattr(self.subnet, "make_readout"):
            self.velocity_readout = self.subnet.make_readout()
            self.velocity_readout.build((h_shape, context_shape))
            h_shape = self.velocity_readout.compute_output_shape((h_shape, context_shape))

        self.velocity_projector = self._build_projector(h_shape, xz_shape[-1], name="velocity_projector")

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
            "steps": self.steps,
            "velocity_head_layers": self.velocity_head_layers,
            "velocity_loss_fn": self.velocity_loss_fn,
            "density_head": self.density_head,
            "boundary_prob": self.boundary_prob,
            "cycle_weight": self.cycle_weight,
            "cycle_mismatch_prob": self.cycle_mismatch_prob,
            "adaptive_loss_power": self.adaptive_loss_power,
            "adaptive_loss_eps": self.adaptive_loss_eps,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)

    def compute_metrics(
        self,
        x: Tensor,
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        training = stage == "training"

        z = self.base_distribution.sample(keras.ops.shape(x)[:-1], seed=self.seed_generator)
        # a fraction of the time pairs is sampled with s = t
        s, t = self._sample_times(x, boundary_prob=self.boundary_prob)

        if self.use_optimal_transport:
            provided_masks = self.mask_handler.collect_mask_kwargs(kwargs)
            if provided_masks:
                names = ", ".join(provided_masks)
                raise ValueError(
                    "Explicit per-example masks cannot be combined with optimal transport because "
                    f"the transport assignment would reorder them ({names}). Use generated masks or "
                    "disable optimal transport."
                )
            z, x, conditions, assignments = optimal_transport(
                z,
                x,
                conditions=conditions,
                seed=self.seed_generator,
                return_assignments=True,
                **self.optimal_transport_kwargs,
            )
            if sample_weight is not None:
                sample_weight = keras.ops.take(sample_weight, assignments, axis=0)

        masks = self.mask_handler.resolve(kwargs, targets=x, conditions=conditions, stage=stage, subnet=self.subnet)

        def masked(value):
            return self.mask_handler.maybe_apply(value, masks.fixed_target_mask)

        z_s = self.mask_handler.maybe_apply((1 - s) * x + s * z, masks.fixed_target_mask, x)
        target_velocity = z - x

        # instantaneous velocity from the velocity head, or the average velocity at s = t
        velocity = self._velocity(z_s, s, conditions, training, masks)

        def average_velocity(z_s, s, t):
            return self._average_velocity(z_s, s, t, conditions, training, masks)

        def average_velocity_and_divergence(z_s, s, t):
            return self._average_velocity_and_divergence(z_s, s, t, conditions, training, masks)

        # total derivative of the average velocity (and divergence) along the path, along which fixed targets do
        # not move; the tangent is cast in case the subnet computes in a lower precision
        tangent = masked(keras.ops.stop_gradient(velocity))
        tangents = (keras.ops.cast(tangent, z_s.dtype), keras.ops.ones_like(s), keras.ops.zeros_like(t))
        if self.density_head:
            (u, divergence), (du_ds, ddivergence_ds) = jvp(average_velocity_and_divergence, (z_s, s, t), tangents)
        else:
            u, du_ds = jvp(average_velocity, (z_s, s, t), tangents)
        predicted_velocity = u - (t - s) * keras.ops.stop_gradient(du_ds)

        # summed over the target dimensions as in [2], for which the adaptive weight is tuned
        loss = keras.ops.shape(x)[-1] * self.loss_fn(masked(target_velocity), masked(predicted_velocity))
        loss = loss * keras.ops.stop_gradient((loss + self.adaptive_loss_eps) ** -self.adaptive_loss_power)

        if self.velocity_blocks:
            loss += self.velocity_loss_fn(masked(target_velocity), masked(velocity))

        if self.density_head:
            # conditional divergence matching of the Jacobian diagonal [4], whose target
            # (z * (v - (z - x)) + 1) / s is weighted by s^2 to stay bounded near the data
            predicted_divergence = divergence - (t - s) * keras.ops.stop_gradient(ddivergence_ds)
            target_divergence = z * (keras.ops.stop_gradient(velocity) - target_velocity) + 1.0
            loss += keras.losses.mean_squared_error(masked(target_divergence), masked(s * predicted_divergence))

        if self.cycle_weight > 0.0:
            s, t = self._sample_times(x)
            z_s = self.mask_handler.maybe_apply((1 - s) * x + s * z, masks.fixed_target_mask, x)

            cycle_conditions = conditions
            if conditions is not None and self.cycle_mismatch_prob > 0.0:
                # the flow map must be invertible for any conditions
                batch_shape = keras.ops.shape(conditions)[:1]
                mismatch = keras.random.uniform(batch_shape, dtype=conditions.dtype, seed=self.seed_generator)
                mismatch = expand_right_as(mismatch < self.cycle_mismatch_prob, conditions)
                cycle_conditions = keras.ops.where(mismatch, keras.ops.roll(conditions, shift=1, axis=0), conditions)

            z_t = self._flow_map(z_s, s, t, cycle_conditions, training, masks)
            z_t = self.mask_handler.maybe_apply(z_t, masks.fixed_target_mask, x)
            z_s_cycle = self._flow_map(z_t, t, s, cycle_conditions, training, masks)

            loss += self.cycle_weight * self.loss_fn(masked(z_s), masked(z_s_cycle))

        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}

    def _sample_times(self, x: Tensor, boundary_prob: float = 0.0) -> tuple[Tensor, Tensor]:
        batch_shape = keras.ops.shape(x)[:1]
        u_s = keras.random.uniform(batch_shape, dtype=x.dtype, seed=self.seed_generator)
        u_t = keras.random.uniform(batch_shape, dtype=x.dtype, seed=self.seed_generator)

        # inverse CDF of F(t) = t^(1/(1+α)), i.e. p(t) ∝ t^(-α/(1+α)); α=0 is uniform
        s = u_s ** (1 + self.time_power_law_alpha)
        t = u_t ** (1 + self.time_power_law_alpha)

        if boundary_prob > 0.0:
            boundary = keras.random.uniform(batch_shape, dtype=x.dtype, seed=self.seed_generator) < boundary_prob
            t = keras.ops.where(boundary, s, t)

        return expand_right_as(s, x), expand_right_as(t, x)

    def velocity(
        self, xz: Tensor, time: float | Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor:
        masks = self.mask_handler.resolve(kwargs, targets=xz, conditions=conditions, subnet=self.subnet)
        return self._velocity(xz, time, conditions, training, masks)

    def _velocity(
        self, xz: Tensor, time: float | Tensor, conditions: Tensor, training: bool, masks: InferenceMasks
    ) -> Tensor:
        if not self.velocity_blocks:
            return self._average_velocity(xz, time, time, conditions, training, masks)

        blocks = [*self.subnet.blocks[: -len(self.velocity_blocks)], *self.velocity_blocks]
        h, context = self._hidden(xz, time, time, conditions, training, masks, blocks)
        if self.velocity_readout is not None:
            h = self.velocity_readout((h, context), training=training)
        out = self.velocity_projector(h)

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            out = self.mask_handler.maybe_apply(out, masks.fixed_target_mask)
        return out

    def _average_velocity(
        self,
        xz: Tensor,
        source_time: float | Tensor,
        target_time: float | Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
    ) -> Tensor:
        time = self._time_pair(source_time, target_time, xz)
        subnet_out = self.subnet((xz, time, conditions), training=training, **masks.subnet_kwargs)
        out = self.output_projector(subnet_out)

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            out = self.mask_handler.maybe_apply(out, masks.fixed_target_mask)
        return out

    def _average_velocity_and_divergence(
        self,
        xz: Tensor,
        source_time: float | Tensor,
        target_time: float | Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
    ) -> tuple[Tensor, Tensor]:
        """The average velocity and the average Jacobian diagonal of the density head, from one subnet pass."""
        h, context = self._hidden(xz, source_time, target_time, conditions, training, masks, self.subnet.blocks)

        u = self.subnet.readout((h, context), training=training) if hasattr(self.subnet, "readout") else h
        u = self.output_projector(u)
        divergence = h if self.divergence_readout is None else self.divergence_readout((h, context), training=training)
        divergence = self.divergence_projector(divergence)

        if not training:
            u = self.mask_handler.maybe_apply(u, masks.fixed_target_mask)
            divergence = self.mask_handler.maybe_apply(divergence, masks.fixed_target_mask)
        return u, divergence

    def _hidden(
        self,
        xz: Tensor,
        source_time: float | Tensor,
        target_time: float | Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
        blocks: list[keras.Layer],
    ) -> tuple[Tensor, Any]:
        """The hidden state after embedding and *blocks*, with the context of the subnet."""
        time = self._time_pair(source_time, target_time, xz)
        h, context = self.subnet.embed((xz, time, conditions), training=training, **masks.subnet_kwargs)
        for block in blocks:
            h = block((h, context), training=training)
        return h, context

    def _flow_map(
        self,
        xz: Tensor,
        source_time: float | Tensor,
        target_time: float | Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
    ) -> Tensor:
        velocity = self._average_velocity(xz, source_time, target_time, conditions, training, masks)
        return xz + (target_time - source_time) * velocity

    def _flow_map_log_det(
        self,
        xz: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        conditions: Tensor,
        training: bool,
        masks: InferenceMasks,
        use_density_head: bool,
    ) -> tuple[Tensor, Tensor]:
        if use_density_head:
            u, divergence = self._average_velocity_and_divergence(
                xz, source_time, target_time, conditions, training, masks
            )
            step = target_time - source_time
            return xz + step * u, step * keras.ops.sum(divergence, axis=-1)

        def f(x):
            return self._flow_map(x, source_time, target_time, conditions, training, masks)

        out, vjp_fn = vjp(f, xz)

        # the vector-Jacobian products with the unit vectors are the rows of the Jacobian
        unit_vectors = keras.ops.eye(xz.shape[-1], dtype=xz.dtype)
        rows = [vjp_fn(keras.ops.broadcast_to(unit_vectors[i], keras.ops.shape(xz)))[0] for i in range(xz.shape[-1])]

        return out, log_abs_det(keras.ops.stack(rows, axis=-2))

    @staticmethod
    def _broadcast_time(time: float | Tensor, xz: Tensor) -> Tensor:
        time = keras.ops.convert_to_tensor(time, dtype=keras.ops.dtype(xz))
        time = expand_right_as(time, xz)
        return keras.ops.broadcast_to(time, keras.ops.shape(xz)[:-1] + (1,))

    @classmethod
    def _time_pair(cls, source_time: float | Tensor, target_time: float | Tensor, xz: Tensor) -> Tensor:
        """The subnet time input ``(s, s - t)``."""
        source_time = cls._broadcast_time(source_time, xz)
        target_time = cls._broadcast_time(target_time, xz)
        return keras.ops.concatenate([source_time, source_time - target_time], axis=-1)

    @staticmethod
    def make_time_schedule(steps: int, rho: float = 7.0, sigma_min: float = 1e-3) -> Tensor:
        """EDM time schedule of ``steps`` steps, decreasing from 1 to 0."""
        fraction = keras.ops.arange(steps + 1, dtype=keras.config.floatx()) / steps
        sigma = (1.0 + fraction * (sigma_min ** (1 / rho) - 1.0)) ** rho
        return (sigma - sigma_min) / (1.0 - sigma_min)

    def _time_schedule(self, kwargs: dict[str, Any]) -> Tensor:
        time_schedule = kwargs.pop("time_schedule", self.time_schedule)
        steps = kwargs.pop("steps", self.steps)
        rho = kwargs.pop("rho", self.rho)
        sigma_min = kwargs.pop("sigma_min", self.sigma_min)

        if time_schedule is None:
            time_schedule = self.make_time_schedule(steps, rho=rho, sigma_min=sigma_min)
        return keras.ops.convert_to_tensor(time_schedule, dtype=keras.config.floatx())

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        masks = self.mask_handler.resolve(kwargs, targets=x, conditions=conditions, subnet=self.subnet)

        # During a forward pass the observed sample supplies fixed values unless
        # the caller provided an explicit value.
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)
        use_density_head = kwargs.pop("use_density_head", self.use_density_head) and self.density_head

        if kwargs.pop("use_instantaneous_flow", self.use_instantaneous_flow):
            if kwargs.get("method", self.integrate_kwargs["method"]) == "glass":
                logging.warning("GLASS only supports sampling. Falling back to tsit5 ODE solver for the forward pass.")
                kwargs = kwargs | {"method": "tsit5"}

            z, log_det = self._integrate_velocity(x, 0.0, 1.0, conditions, density, training, masks, kwargs)
        else:
            times = keras.ops.flip(self._time_schedule(kwargs))
            z, log_det = self._compose_flow_maps(x, times, conditions, density, training, masks, use_density_head)

        if density:
            return z, self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) + log_det
        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        masks = self.mask_handler.resolve(kwargs, targets=z, conditions=conditions, subnet=self.subnet)
        z = self.mask_handler.maybe_apply(z, masks.fixed_target_mask, masks.fixed_target_value)
        use_density_head = kwargs.pop("use_density_head", self.use_density_head) and self.density_head

        if kwargs.pop("use_instantaneous_flow", self.use_instantaneous_flow):
            x, log_det = self._integrate_velocity(z, 1.0, 0.0, conditions, density, training, masks, kwargs)
        else:
            times = self._time_schedule(kwargs)
            x, log_det = self._compose_flow_maps(z, times, conditions, density, training, masks, use_density_head)

        if density:
            return x, self.base_distribution.log_prob(z, mask=masks.fixed_target_mask) - log_det
        return x

    def _compose_flow_maps(
        self,
        x: Tensor,
        times: Tensor,
        conditions: Tensor,
        density: bool,
        training: bool,
        masks: InferenceMasks,
        use_density_head: bool,
    ) -> tuple[Tensor, Tensor]:
        """Applies the flow maps between consecutive ``times``, returning the state and the summed log-determinants."""
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1], dtype=x.dtype)
        for i in range(len(times) - 1):
            if density:
                x, step_log_det = self._flow_map_log_det(
                    x, times[i], times[i + 1], conditions, training, masks, use_density_head
                )
                log_det += step_log_det
            else:
                x = self._flow_map(x, times[i], times[i + 1], conditions, training, masks)
        return x, log_det

    def _integrate_velocity(
        self,
        xz: Tensor,
        start_time: float,
        stop_time: float,
        conditions: Tensor,
        density: bool,
        training: bool,
        masks: InferenceMasks,
        kwargs: dict[str, Any],
    ) -> tuple[Tensor, Tensor | None]:
        """Integrates the instantaneous velocity, returning the state and, with ``density``, the trace integral."""
        seed = resolve_seed(kwargs.pop("seed", None), self.seed_generator)
        hutchinson_steps = kwargs.pop("hutchinson_steps", None)

        # Build integrate kwargs: instance config -> call-time overrides
        integrate_kwargs = self.integrate_kwargs | self.mask_handler.without_mask_kwargs(kwargs)

        if density and integrate_kwargs["method"] == "glass":
            logging.warning("GLASS is not supported for density computation. Falling back to tsit5 ODE solver.")
            integrate_kwargs["method"] = "tsit5"

        if not density:

            def deltas(time, xz):
                return {"xz": self._velocity(xz, time, conditions, training, masks)}

            if integrate_kwargs["method"] == "glass":
                project_state, project_updates = self.mask_handler.integrator_callbacks(masks, replacement=xz)
                state = integrate_stochastic(
                    drift_fn=deltas,
                    diffusion_fn=None,
                    state={"xz": xz},
                    start_time=start_time,
                    stop_time=stop_time,
                    noise_schedule="flow_matching",
                    seed=seed,
                    state_projection_fn=project_state,
                    state_update_projection_fn=project_updates,
                    **integrate_kwargs,
                )
            else:
                state = integrate(deltas, {"xz": xz}, start_time=start_time, stop_time=stop_time, **integrate_kwargs)
            return state["xz"], None

        # one probe per solve, fixed over time, keeps the augmented ODE deterministic
        projectors = hutchinson_projectors(xz, hutchinson_steps, seed=seed)

        def deltas(time, xz):
            def f(x):
                return self._velocity(x, time, conditions, training, masks)

            v, trace = jacobian_trace(f, xz, projectors=projectors, return_output=True)
            return {"xz": v, "trace": keras.ops.expand_dims(trace, axis=-1)}

        state = {"xz": xz, "trace": keras.ops.zeros(keras.ops.shape(xz)[:-1] + (1,), dtype=xz.dtype)}
        state = integrate(deltas, state, start_time=start_time, stop_time=stop_time, **integrate_kwargs)

        return state["xz"], keras.ops.squeeze(state["trace"], axis=-1)
