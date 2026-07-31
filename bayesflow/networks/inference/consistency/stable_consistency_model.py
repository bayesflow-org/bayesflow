from math import pi

import keras
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import (
    expand_right_as,
    expand_right_to,
    find_network,
    filter_kwargs,
    jvp,
    layer_kwargs,
    logging,
    MaskName,
    maybe_mask_tensor,
    resolve_seed,
    sample_input_masks,
    weighted_mean,
)
from bayesflow.utils.serialization import serializable, serialize

from ...inference import InferenceNetwork
from ...defaults import TIME_MLP_DEFAULTS, DIFFUSION_TRANSFORMER_DEFAULTS, WEIGHT_MLP_DEFAULTS


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
        sample training times (``P_mean`` in [1]). Default is ``-1.0``.
    noise_dist_std : float
        Standard deviation of that proposal distribution (``P_std`` in [1]).
        Default is ``1.6``.
    tangent_norm_eps : float
        Small constant added to the tangent norm when normalizing the training
        target for stability (``c`` in [1]). Default is ``0.1``.
    steps : int
        Default number of steps used by the multistep sampler. Can be overridden
        per call by passing ``steps=`` to sampling. Default is ``15``.
    rho : float
        Exponent controlling the curvature of the time discretization schedule
        used at sampling time. Can be overridden per call by passing ``rho=`` to
        sampling. Default is ``3.5``.
    subnet_kwargs : dict[str, any], optional
        Keyword arguments passed to the constructor of the chosen *subnet*
        (e.g., number of hidden units, activation functions, or dropout settings).
    weight_mlp_kwargs : dict[str, any], optional
        Keyword arguments for an auxiliary MLP used to generate weights within the
        consistency model (e.g., depth, hidden sizes, non-linearity choices).
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
    _SUBNET_MASK_KEYS = {
        "attention_mask",
        MaskName.FIXED_TARGET,
        MaskName.INFER_TARGET,
        MaskName.OBSERVED_CONDITION,
    }

    def __init__(
        self,
        subnet: str | type | keras.Layer = "time_mlp",
        sigma: float = 1.0,
        noise_dist_mean: float = -1.0,
        noise_dist_std: float = 1.6,
        tangent_norm_eps: float = 0.1,
        steps: int = 15,
        rho: float = 3.5,
        subnet_kwargs: dict[str, any] = None,
        weight_mlp_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(base_distribution="normal", **kwargs)

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "time_mlp":
            subnet_kwargs = TIME_MLP_DEFAULTS | subnet_kwargs
        if subnet == "diffusion_transformer":
            subnet_kwargs = DIFFUSION_TRANSFORMER_DEFAULTS | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)
        self._subnet_mask_keys = set(filter_kwargs({k: None for k in self._SUBNET_MASK_KEYS}, self.subnet.call).keys())

        self.subnet_projector = None

        weight_mlp_kwargs = weight_mlp_kwargs or {}
        weight_mlp_kwargs = WEIGHT_MLP_DEFAULTS | weight_mlp_kwargs
        self.weight_fn = find_network("mlp", **weight_mlp_kwargs)

        self.weight_fn_projector = keras.layers.Dense(
            units=1, bias_initializer="zeros", kernel_initializer="zeros", name="weight_fn_projector"
        )

        self.sigma = sigma
        self.noise_dist_mean = noise_dist_mean
        self.noise_dist_std = noise_dist_std
        self.tangent_norm_eps = tangent_norm_eps
        self.steps = steps
        self.rho = rho
        self.fixed_target_prob = kwargs.get("fixed_target_prob", 0.0)
        self.missing_target_prob = kwargs.get("missing_target_prob", 0.0)
        self.missing_conditions_prob = kwargs.get("missing_conditions_prob", 0.0)
        self.seed_generator = keras.random.SeedGenerator()

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "sigma": self.sigma,
            "noise_dist_mean": self.noise_dist_mean,
            "noise_dist_std": self.noise_dist_std,
            "tangent_norm_eps": self.tangent_norm_eps,
            "steps": self.steps,
            "rho": self.rho,
            "fixed_target_prob": self.fixed_target_prob,
            "missing_target_prob": self.missing_target_prob,
            "missing_conditions_prob": self.missing_conditions_prob,
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

        # construct input shape for subnet and subnet projector
        time_shape = (xz_shape[0], 1)  # same batch dims, 1 feature
        self.subnet.build((xz_shape, time_shape, conditions_shape))
        out_shape = self.subnet.compute_output_shape((xz_shape, time_shape, conditions_shape))
        if out_shape[-1] != xz_shape[-1]:
            self.subnet_projector = keras.layers.Dense(
                units=xz_shape[-1],
                bias_initializer="zeros",
                name="subnet_projector",
            )
        else:
            self.subnet_projector = keras.layers.Identity()
        self.subnet_projector.build(out_shape)

        # input shape for weight function and projector
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
        seed = resolve_seed(kwargs.pop("seed", None)) or self.seed_generator
        subnet_kwargs = self._collect_mask_kwargs(self._subnet_mask_keys, kwargs)

        steps = kwargs.get("steps", self.steps)
        rho = kwargs.get("rho", self.rho)

        # noise distribution has variance sigma
        x = keras.ops.copy(z) * self.sigma
        discretized_time = keras.ops.flip(self._discretize_time(steps, rho=rho), axis=-1)
        t = keras.ops.full((*keras.ops.shape(x)[:-1], 1), discretized_time[0], dtype=x.dtype)

        # Apply user-provided target mask if available
        fixed_target_mask = kwargs.get(MaskName.FIXED_TARGET, None)
        targets_fixed = kwargs.get(MaskName.FIXED_TARGET_VALUE, None)
        if fixed_target_mask is not None:
            fixed_target_mask = keras.ops.broadcast_to(fixed_target_mask, keras.ops.shape(x))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(x))
            x = maybe_mask_tensor(x, mask=fixed_target_mask, replacement=targets_fixed)

        # apply consistency function at t_1
        x = self.consistency_function(x, t, conditions=conditions, **subnet_kwargs)
        x = maybe_mask_tensor(x, mask=fixed_target_mask, replacement=targets_fixed)

        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=seed)
            t = keras.ops.full_like(t, discretized_time[n])
            x_n = ops.cos(t) * x + ops.sin(t) * noise
            x_n = maybe_mask_tensor(x_n, mask=fixed_target_mask, replacement=targets_fixed)
            x = self.consistency_function(x_n, t, conditions=conditions, **subnet_kwargs)
            x = maybe_mask_tensor(x, mask=fixed_target_mask, replacement=targets_fixed)
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
        subnet_out = self.subnet((x / self.sigma, t, conditions), training=training, **kwargs)
        f = self.subnet_projector(subnet_out)
        out = ops.cos(t) * x - ops.sin(t) * self.sigma * f
        return out

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, stage: str = "training", sample_weight: Tensor = None, **kwargs
    ) -> dict[str, Tensor]:
        training = stage == "training"

        subnet_kwargs = self._collect_mask_kwargs(self._subnet_mask_keys, kwargs)

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

        # Generate target / condition / missingness masks
        mask_x, loss_mask, subnet_kwargs = sample_input_masks(
            self.subnet,
            x,
            conditions,
            subnet_kwargs,
            training,
            fixed_target_prob=self.fixed_target_prob,
            missing_target_prob=self.missing_target_prob,
            missing_conditions_prob=self.missing_conditions_prob,
            seed_generator=self.seed_generator,
        )
        xt = maybe_mask_tensor(xt, mask=mask_x, replacement=x)

        # calculate estimator for dx_t/dt
        dxtdt = ops.cos(t) * z - ops.sin(t) * x
        dxtdt = maybe_mask_tensor(dxtdt, mask=mask_x)  # replace with zeros

        r = 1.0  # TODO: if consistency distillation training (not supported yet) is unstable, add schedule here

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
        g = -(ops.cos(t) ** 2) * (self.sigma * teacher_output - dxtdt) - r * ops.cos(t) * ops.sin(t) * (
            xt + self.sigma * cos_sin_dFdt
        )

        # apply normalization to stabilize training
        g = g / (ops.norm(g, axis=-1, keepdims=True) + self.tangent_norm_eps)

        # compute adaptive weights and calculate loss
        w = self.weight_fn_projector(self.weight_fn(expand_right_to(t_, 2)))

        D = ops.shape(x)[-1]

        loss = ops.mean(
            ops.reshape((loss_mask * (student_out - teacher_output - g) ** 2), (ops.shape(teacher_output)[0], -1)),
            axis=-1,
        )
        loss = (ops.exp(w) / D) * loss - w
        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}
