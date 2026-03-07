from collections.abc import Sequence

import keras

from bayesflow.distributions import Distribution
from bayesflow.types import Shape, Tensor
from bayesflow.utils import (
    logging,
    expand_right_as,
    find_network,
    integrate,
    jacobian_trace,
    layer_kwargs,
    optimal_transport,
    randomly_mask_conditions,
    weighted_mean,
)
from bayesflow.utils.serialization import serialize, serializable

from ...inference_network import InferenceNetwork
from ...defaults import TIME_MLP_DEFAULTS, FLOW_MATCHING_INTEGRATE_DEFAULTS, OPTIMAL_TRANSPORT_DEFAULTS


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
    subnet : str, type, or keras.Layer, optional
        A neural network type for the flow matching model, will be instantiated
        using *subnet_kwargs*.  If a string is provided, it should be a registered
        name (e.g., ``"time_mlp"``).  If a type or ``keras.Layer`` is provided, it
        will be directly instantiated with the given *subnet_kwargs*.  Any subnet
        must accept a tuple of tensors ``(target, time, conditions)``.
    base_distribution : str or Distribution, optional
        The base probability distribution from which samples are drawn.
        Default is ``"normal"``.
    use_optimal_transport : bool, optional
        Whether to apply optimal transport for improved training stability.
        Default is ``False``.  Note: this will increase training time by
        approximately 2.5×, but may lead to faster inference.
    loss_fn : str or keras.Loss, optional
        The loss function used for training.  Default is ``"mse"``.
    integrate_kwargs : dict[str, any], optional
        Additional keyword arguments for the ODE integrator used at inference time.
    optimal_transport_kwargs : dict[str, any], optional
        Additional keyword arguments for configuring optimal transport.
    subnet_kwargs : dict[str, any], optional
        Keyword arguments passed to the subnet constructor or used to update the
        default MLP settings.
    time_power_law_alpha : float, optional
        Changes the distribution of sampled times during training.  Time is sampled
        from a power-law distribution ``p(t) ~ t^(1/(1+alpha))``, where
        ``alpha`` is the provided value.  Default is 0 (uniform sampling).
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
        integrate_kwargs: dict[str, any] = None,
        optimal_transport_kwargs: dict[str, any] = None,
        subnet_kwargs: dict[str, any] = None,
        time_power_law_alpha: float = 0.0,
        drop_cond_prob: float = 0.0,
        drop_target_prob: float = 0.0,
        **kwargs,
    ):
        super().__init__(base_distribution, **kwargs)

        self.use_optimal_transport = use_optimal_transport

        self.integrate_kwargs = FLOW_MATCHING_INTEGRATE_DEFAULTS | (integrate_kwargs or {})
        self.optimal_transport_kwargs = OPTIMAL_TRANSPORT_DEFAULTS | (optimal_transport_kwargs or {})

        self.loss_fn = keras.losses.get(loss_fn)
        self.time_power_law_alpha = float(time_power_law_alpha)
        if self.time_power_law_alpha <= -1.0:
            raise ValueError("'time_power_law_alpha' must be greater than -1.0.")

        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "time_mlp":
            subnet_kwargs = TIME_MLP_DEFAULTS | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self.output_projector = None
        self.drop_cond_prob = drop_cond_prob
        self.unconditional_mode = False
        self.drop_target_prob = drop_target_prob

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)

        self.output_projector = keras.layers.Dense(
            units=xz_shape[-1],
            bias_initializer="zeros",
            name="output_projector",
        )

        # construct input shape for subnet and subnet projector
        time_shape = tuple(xz_shape[:-1]) + (1,)  # same batch/sequence dims, 1 feature
        self.subnet.build((xz_shape, time_shape, conditions_shape))
        out_shape = self.subnet.compute_output_shape((xz_shape, time_shape, conditions_shape))

        self.output_projector.build(out_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "base_distribution": self.base_distribution,
            "use_optimal_transport": self.use_optimal_transport,
            "loss_fn": self.loss_fn,
            "integrate_kwargs": self.integrate_kwargs,
            "optimal_transport_kwargs": self.optimal_transport_kwargs,
            "time_power_law_alpha": self.time_power_law_alpha,
            "drop_cond_prob": self.drop_cond_prob,
            "drop_target_prob": self.drop_target_prob,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)

    def _generate_target_mask(self, shape: Shape, seed=None) -> Tensor:
        """Generate a random binary mask for target dropout.

        Returns a mask where 1.0 = keep, 0.0 = drop, based on drop_target_prob.
        If drop_target_prob is 0, returns 1.0 (keep all).
        """
        from keras import ops

        if self.drop_target_prob <= 0:
            return 1.0

        random_vals = keras.random.uniform(shape=shape, dtype=keras.ops.dtype(1.0), seed=seed)
        return ops.cast(random_vals > self.drop_target_prob, dtype=keras.ops.dtype(1.0))

    def _apply_target_mask(self, data: Tensor, mask: Tensor | None = None, replacement: Tensor = None) -> Tensor:
        """Apply a target mask to data.

        Parameters
        ----------
        data : Tensor
            The data to mask
        mask : Tensor or None, optional
            Binary mask where 1.0 = keep, 0.0 = drop. If None, returns data unchanged.
        replacement : Tensor, optional
            Values to use where mask is 0. If None, uses original data.

        Returns
        -------
        Tensor
            Masked data: mask * data + (1 - mask) * replacement, or data if mask is None
        """
        if mask is None:
            return data

        if replacement is None:
            replacement = data
        return mask * data + (1 - mask) * replacement

    def velocity(
        self, xz: Tensor, time: float | Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor:
        # Extract subnet masks from kwargs
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        time = keras.ops.convert_to_tensor(time, dtype=keras.ops.dtype(xz))
        time = expand_right_as(time, xz)
        time = keras.ops.broadcast_to(time, keras.ops.shape(xz)[:-1] + (1,))
        subnet_out = self.subnet((xz, time, conditions), training=training, **subnet_kwargs)
        out = self.output_projector(subnet_out)

        # Zero out velocity where target is fixed (during inference only)
        target_mask = kwargs.get("target_mask", None)
        if self.drop_target_prob > 0 and not training and target_mask is not None:
            out = self._apply_target_mask(out, mask=target_mask, replacement=keras.ops.zeros_like(out))
        return out

    def _velocity_trace(
        self,
        xz: Tensor,
        time: Tensor,
        conditions: Tensor = None,
        max_steps: int = None,
        training: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        def f(x):
            return self.velocity(x, time=time, conditions=conditions, training=training, **kwargs)

        v, trace = jacobian_trace(f, xz, max_steps=max_steps, seed=self.seed_generator, return_output=True)

        return v, keras.ops.expand_dims(trace, axis=-1)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Remaining kwargs go to the integrator
        integrate_kwargs = self.integrate_kwargs | {
            k: v for k, v in kwargs.items() if k in FLOW_MATCHING_INTEGRATE_DEFAULTS
        }

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if self.drop_target_prob > 0 and target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(x))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(x))
            x = self._apply_target_mask(x, target_mask, replacement=targets_fixed)

        if self.unconditional_mode and conditions is not None:
            conditions = keras.ops.zeros_like(conditions)
            logging.info("Condition masking is applied: conditions are set to zero.")

        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training, **kwargs)
                return {"xz": v, "trace": trace}

            state = {"xz": x, "trace": keras.ops.zeros(keras.ops.shape(x)[:-1] + (1,), dtype=keras.ops.dtype(x))}
            state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **integrate_kwargs)

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z) + keras.ops.squeeze(state["trace"], axis=-1)

            return z, log_density

        def deltas(time, xz):
            return {"xz": self.velocity(xz, time=time, conditions=conditions, training=training, **kwargs)}

        state = {"xz": x}
        state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **integrate_kwargs)

        z = state["xz"]

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Remaining kwargs go to the integrator
        integrate_kwargs = self.integrate_kwargs | {
            k: v for k, v in kwargs.items() if k in FLOW_MATCHING_INTEGRATE_DEFAULTS
        }

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if self.drop_target_prob > 0 and target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(z))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(z))
            z = self._apply_target_mask(z, target_mask, replacement=targets_fixed)

        if self.unconditional_mode and conditions is not None:
            conditions = keras.ops.zeros_like(conditions)
            logging.info("Condition masking is applied: conditions are set to zero.")

        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training, **kwargs)
                return {"xz": v, "trace": trace}

            state = {"xz": z, "trace": keras.ops.zeros(keras.ops.shape(z)[:-1] + (1,), dtype=keras.ops.dtype(z))}
            state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **integrate_kwargs)

            x = state["xz"]
            log_density = self.base_distribution.log_prob(z) - keras.ops.squeeze(state["trace"], axis=-1)

            return x, log_density

        def deltas(time, xz):
            return {"xz": self.velocity(xz, time=time, conditions=conditions, training=training, **kwargs)}

        state = {"xz": z}
        state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **integrate_kwargs)

        x = state["xz"]

        return x

    def compute_metrics(
        self,
        x: Tensor | Sequence[Tensor],
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        # Extract subnet masks from kwargs
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        if isinstance(x, Sequence):
            # already pre-configured
            x0, x1, t, x, target_velocity = x
        else:
            x1 = x
            x0 = self.base_distribution.sample(keras.ops.shape(x1)[:-1])

            if self.use_optimal_transport:
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

        if self.drop_cond_prob > 0 and conditions is not None:
            conditions = randomly_mask_conditions(conditions, self.drop_cond_prob, self.seed_generator)

        # Generate target dropout mask
        mask_x = self._generate_target_mask(keras.ops.shape(x), seed=self.seed_generator)
        x = self._apply_target_mask(x, mask=mask_x, replacement=x1)

        predicted_velocity = self.velocity(
            x, time=t, conditions=conditions, training=stage == "training", **subnet_kwargs
        )

        loss = self.loss_fn(mask_x * target_velocity, mask_x * predicted_velocity)
        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}
