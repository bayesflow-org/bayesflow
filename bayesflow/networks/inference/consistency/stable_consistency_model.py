from typing import Any
from math import pi

import keras
from keras import ops

from bayesflow.masking import MaskHandler
from bayesflow.types import Tensor
from bayesflow.utils import (
    expand_right_as,
    expand_right_to,
    find_network,
    jvp,
    logging,
    resolve_seed,
    weighted_mean,
)
from bayesflow.utils.serialization import serializable, serialize
from ...inference import InferenceNetwork
from ...defaults import CONSISTENCY_MODEL_DEFAULTS, DIFFUSION_TRANSFORMER_DEFAULTS, WEIGHT_MLP_DEFAULTS


@serializable("bayesflow.networks")
class StableConsistencyModel(InferenceNetwork):
    """Stable consistency model (sCM) for simulation-based inference.

    Implements the simple, stable, and scalable Consistency Model with
    continuous-time Consistency Training (CT) as described in [1].  The sampling
    procedure is taken from [2].

    Note: With the PyTorch backend on CPU, ``subnet="diffusion_transformer"`` is not supported because PyTorch CPU
    scaled-dot-product attention lacks forward-mode AD support.

    Parameters
    ----------
    subnet : str, type, or keras.Layer
        The neural network architecture used for the consistency model.  If a
        string is provided, it should be a registered name (e.g., ``"time_mlp"``).
        If a type or ``keras.Layer`` is provided, it will be directly instantiated
        with the given *subnet_kwargs*.  Any subnet must accept a tuple of tensors
        ``(target, time, conditions)``.  Default is ``"time_mlp"``.
    sigma : float
        Standard deviation of the target distribution for the consistency loss.
        Controls the scale of the noise injected during training.  Default is 1.0.
    noise_dist_mean : float
        Mean of the log-normal proposal distribution over noise levels used to
        sample training times (``P_mean`` in [1]). Default is ``-1.1``.
    noise_dist_std : float
        Standard deviation of that proposal distribution (``P_std`` in [1]).
        Default is ``2.0``.
    tangent_norm_eps : float
        Small constant added to the tangent norm when normalizing the training
        target for stability (``c`` in [1]). Default is ``0.1``.
    r_scale : float
        Scale of the second stable tangent term in the continuous-time
        consistency target. Default is ``1.0``.
    steps : int
        Default number of steps used by the multistep sampler. Can be overridden
        per call by passing ``steps=`` to sampling. Default is ``15``.
    rho : float
        Exponent controlling the curvature of the time discretization schedule
        used at sampling time. Can be overridden per call by passing ``rho=`` to
        sampling. Default is ``3.5``.
    subnet_kwargs : dict[str, Any], optional
        Keyword arguments passed to the constructor of the chosen *subnet*
        (e.g., number of hidden units, activation functions, or dropout settings).
    weight_mlp_kwargs : dict[str, Any], optional
        Keyword arguments for an auxiliary MLP used to generate weights within the
        consistency model (e.g., depth, hidden sizes, non-linearity choices).
    mask_handler : ~bayesflow.masking.MaskHandler, optional
        Handler used to generate and route masks. If ``None``, a default mask handler is created.
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``
        (e.g., ``name``, ``dtype``, or ``trainable``).

    References
    ----------
    [1] Lu, C., & Song, Y. (2024). Simplifying, Stabilizing and Scaling
        Continuous-Time Consistency Models. arXiv:2410.11081.
    [2] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023). Consistency
        Models. arXiv:2303.01469.
    """

    EPS_WARN = 0.1

    def __init__(
        self,
        subnet: str | type | keras.Layer = "time_mlp",
        sigma: float = 1.0,
        noise_dist_mean: float = -1.1,
        noise_dist_std: float = 2.0,
        tangent_norm_eps: float = 0.1,
        r_scale: float = 1.0,
        steps: int = 15,
        rho: float = 3.5,
        subnet_kwargs: dict[str, Any] = None,
        weight_mlp_kwargs: dict[str, Any] = None,
        mask_handler: MaskHandler | None = None,
        **kwargs,
    ):
        super().__init__(base_distribution="normal", mask_handler=mask_handler, **kwargs)

        subnet_defaults = {
            "time_mlp": CONSISTENCY_MODEL_DEFAULTS,
            "diffusion_transformer": DIFFUSION_TRANSFORMER_DEFAULTS,
        }
        self.subnet = find_network(subnet, **(subnet_defaults.get(subnet, {}) | (subnet_kwargs or {})))

        self.subnet_projector = None

        self.weight_mlp_kwargs = WEIGHT_MLP_DEFAULTS | (weight_mlp_kwargs or {})
        self.weight_fn = find_network("mlp", **self.weight_mlp_kwargs)

        self.weight_fn_projector = keras.layers.Dense(
            units=1, bias_initializer="zeros", kernel_initializer="zeros", name="weight_fn_projector"
        )

        self.sigma = sigma
        self.noise_dist_mean = noise_dist_mean
        self.noise_dist_std = noise_dist_std
        self.tangent_norm_eps = tangent_norm_eps
        self.r_scale = r_scale
        self.steps = steps
        self.rho = rho

    def get_config(self):
        base_config = super().get_config()
        config = {
            "subnet": self.subnet,
            "sigma": self.sigma,
            "noise_dist_mean": self.noise_dist_mean,
            "noise_dist_std": self.noise_dist_std,
            "tangent_norm_eps": self.tangent_norm_eps,
            "r_scale": self.r_scale,
            "steps": self.steps,
            "rho": self.rho,
            "weight_mlp_kwargs": self.weight_mlp_kwargs,
        }

        return base_config | serialize(config)

    @staticmethod
    def _discretize_time(num_steps: int, rho: float):
        t = keras.ops.linspace(0.0, pi / 2, num_steps + 1)[1:]
        times = keras.ops.exp((t - pi / 2) * rho) * pi / 2

        # if rho is set too low, bad schedules can occur
        if times[0] > StableConsistencyModel.EPS_WARN:
            logging.warning("Warning: The last time step is large.")
            logging.warning(f"Increasing rho (was {rho}) or n_steps (was {num_steps}) might improve results.")
        return times

    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)
        self.subnet_projector = self._build_subnet_and_projector(xz_shape, conditions_shape)

        # input shape for weight function and its projector
        input_shape = (xz_shape[0], 1)
        self.weight_fn.build(input_shape)
        input_shape = self.weight_fn.compute_output_shape(input_shape)
        self.weight_fn_projector.build(input_shape)

    def _forward(self, x: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        # Consistency Models only learn the direction from noise distribution
        # to target distribution, so we cannot implement this function.
        raise NotImplementedError("Consistency Models are not invertible")

    def _inverse(self, z: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        """Generate random draws from the approximate target distribution
        using the multistep sampling algorithm from [2], Algorithm 1.

        Parameters
        ----------
        z           : Tensor
            Samples from a standard normal distribution
        conditions  : Tensor, optional, default: None
            Conditions for an approximate conditional distribution
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments. Pass `steps` and `rho` to override the
            constructor defaults for the number of sampling steps and the time
            discretization. Subnet-related kwargs (e.g., masks) are passed to the subnet.

        Returns
        -------
        x            : Tensor
            The approximate samples
        """
        seed = resolve_seed(kwargs.pop("seed", None), self.seed_generator)
        masks = self.mask_handler.resolve(
            kwargs,
            targets=z,
            conditions=conditions,
            subnet=self.subnet,
        )
        subnet_kwargs = masks.subnet_kwargs

        steps = kwargs.get("steps", self.steps)
        rho = kwargs.get("rho", self.rho)

        # noise distribution has variance sigma
        x = keras.ops.copy(z) * self.sigma
        discretized_time = keras.ops.flip(self._discretize_time(steps, rho=rho), axis=-1)
        t = keras.ops.full((*keras.ops.shape(x)[:-1], 1), discretized_time[0], dtype=x.dtype)

        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, masks.fixed_target_value)

        # apply consistency function at t_1
        x = self._consistency_function(
            x,
            t,
            conditions=conditions,
            training=False,
            subnet_kwargs=subnet_kwargs,
        )
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, masks.fixed_target_value)

        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=seed) * self.sigma
            t = keras.ops.full_like(t, discretized_time[n])
            x_n = ops.cos(t) * x + ops.sin(t) * noise
            x_n = self.mask_handler.maybe_apply(x_n, masks.fixed_target_mask, masks.fixed_target_value)
            x = self._consistency_function(
                x_n,
                t,
                conditions=conditions,
                training=False,
                subnet_kwargs=subnet_kwargs,
            )
            x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, masks.fixed_target_value)
        return x

    def consistency_function(
        self, x: Tensor, t: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor:
        """Compute consistency function at time t.

        Parameters
        ----------
        x           : Tensor
            Input vector
        t           : Tensor
            Vector of time samples in [0, pi/2]
        conditions  : Tensor
            The conditioning vector
        training    : bool
            Flag to control whether the inner network operates in training or test mode
        **kwargs    : dict, optional
            Additional keyword arguments to pass to the subnet.
        """
        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            subnet=self.subnet,
        )
        replacement = masks.fixed_target_value if masks.fixed_target_value is not None else x
        x = self.mask_handler.maybe_apply(x, masks.fixed_target_mask, replacement)
        out = self._consistency_function(
            x,
            t,
            conditions=conditions,
            training=training,
            subnet_kwargs=masks.subnet_kwargs,
        )
        return self.mask_handler.maybe_apply(out, masks.fixed_target_mask, replacement)

    def _consistency_function(
        self,
        x: Tensor,
        t: Tensor,
        conditions: Tensor,
        training: bool,
        subnet_kwargs: dict,
    ) -> Tensor:
        subnet_out = self.subnet((x / self.sigma, t, conditions), training=training, **subnet_kwargs)
        f = self.subnet_projector(subnet_out)
        out = ops.cos(t) * x - ops.sin(t) * self.sigma * f
        return out

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, stage: str = "training", sample_weight: Tensor = None, **kwargs
    ) -> dict[str, Tensor]:
        training = stage == "training"

        # generate noise vector
        z = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator) * self.sigma

        # sample time
        tau = (
            keras.random.normal(keras.ops.shape(x)[:1], dtype=keras.ops.dtype(x), seed=self.seed_generator)
            * self.noise_dist_std
            + self.noise_dist_mean
        )
        t_ = ops.arctan(ops.exp(tau) / self.sigma)
        t = expand_right_as(t_, x)

        # generate noisy sample
        xt = ops.cos(t) * x + ops.sin(t) * z

        masks = self.mask_handler.resolve(
            kwargs,
            targets=x,
            conditions=conditions,
            stage=stage,
            subnet=self.subnet,
        )
        subnet_kwargs = masks.subnet_kwargs
        xt = self.mask_handler.maybe_apply(xt, masks.fixed_target_mask, x)

        # calculate estimator for dx_t/dt
        dxtdt = ops.cos(t) * z - ops.sin(t) * x
        dxtdt = self.mask_handler.maybe_apply(dxtdt, masks.fixed_target_mask)

        def f_teacher(x, t):
            o = self.subnet((x, t, conditions), training=training, **subnet_kwargs)
            return self.subnet_projector(o)

        primals = (xt / self.sigma, t)
        tangents = (
            ops.cos(t) * ops.sin(t) * dxtdt,
            ops.cos(t) * ops.sin(t) * self.sigma,
        )

        teacher_output, cos_sin_dFdt = jvp(f_teacher, primals, tangents, return_output=True)
        teacher_output = ops.stop_gradient(teacher_output)
        cos_sin_dFdt = ops.stop_gradient(cos_sin_dFdt)

        # calculate output of the network
        subnet_out = self.subnet((xt / self.sigma, t, conditions), training=training, **subnet_kwargs)
        student_out = self.subnet_projector(subnet_out)

        # calculate the tangent
        g = -(ops.cos(t) ** 2) * (self.sigma * teacher_output - dxtdt) - self.r_scale * (
            ops.cos(t) * ops.sin(t) * xt + cos_sin_dFdt
        )

        # Fixed-coordinate tangents must not change the normalization of the
        # active-coordinate training target.
        g = self.mask_handler.maybe_apply(g, masks.fixed_target_mask)

        # apply normalization to stabilize training
        g = g / (ops.norm(g, axis=-1, keepdims=True) + self.tangent_norm_eps)

        # compute adaptive weights and calculate loss
        w = ops.squeeze(self.weight_fn_projector(self.weight_fn(expand_right_to(t_, 2))), axis=-1)

        # the mean over dimensions is the ||.||^2 / D of [1]
        loss = ops.mean(
            ops.reshape(
                self.mask_handler.maybe_apply((student_out - teacher_output - g) ** 2, masks.fixed_target_mask),
                (ops.shape(teacher_output)[0], -1),
            ),
            axis=-1,
        )
        loss = ops.exp(w) * loss - w
        if masks.fixed_target_mask is not None:
            active = ops.any(
                ops.cast(ops.reshape(masks.fixed_target_mask, (ops.shape(x)[0], -1)), "bool"),
                axis=-1,
            )
            # an all-fixed example contributes neither residual nor ``-w`` term
            loss = loss * ops.cast(active, ops.dtype(loss))
        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}
