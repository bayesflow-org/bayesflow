from math import pi

import keras
from keras import ops

from bayesflow.networks import MLP
from bayesflow.types import Tensor
from bayesflow.utils import logging, jvp, find_network, expand_right_as, expand_right_to, layer_kwargs, tensor_utils
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks import InferenceNetwork
from bayesflow.networks.embeddings import FourierEmbedding


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class StableConsistencyModel(InferenceNetwork):
    """(IN) Implements an sCM (simple, stable, and scalable Consistency Model) with continuous-time Consistency Training
    (CT) as described in [1]. The sampling procedure is taken from [2].

    [1] Lu, C., & Song, Y. (2024).
    Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models
    arXiv preprint arXiv:2410.11081

    [2] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023).
    Consistency Models. arXiv preprint arXiv:2303.01469
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    WEIGHT_MLP_DEFAULT_CONFIG = {
        "widths": (256,),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": False,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    EPS_WARN = 0.1

    def __init__(
        self,
        subnet: str | type | keras.Layer = "mlp",
        sigma: float = 1.0,
        subnet_kwargs: dict[str, any] = None,
        weight_mlp_kwargs: dict[str, any] = None,
        embedding_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        """Creates an instance of an sCM to be used for consistency training (CT).

        Parameters
        ----------
        subnet : str, type, or keras.Layer, optional, default="mlp"
            The neural network architecture used for the consistency model.
            If a string is provided, it should be a registered name (e.g., "mlp").
            If a type or keras.Layer is provided, it will be directly instantiated
            with the given ``subnet_kwargs``.
        sigma : float, optional, default=1.0
            Standard deviation of the target distribution for the consistency loss.
            Controls the scale of the noise injected during training.
        subnet_kwargs : dict[str, any], optional, default=None
            Keyword arguments passed to the constructor of the chosen ``subnet``. For example, number of hidden units,
            activation functions, or dropout settings.
        weight_mlp_kwargs : dict[str, any], optional, default=None
            Keyword arguments for an auxiliary MLP used to generate weights within the consistency model. Typically
            includes depth, hidden sizes, and non-linearity choices.
        embedding_kwargs : dict[str, any], optional, default=None
            Keyword arguments for the time embedding layer(s) used in the model
        concatenate_subnet_input: bool, optional
            Flag for advanced users to control whether all inputs to the subnet should be concatenated
            into a single vector or passed as separate arguments. If set to False, the subnet
            must accept three separate inputs: 'x' (noisy parameters), 't' (log signal-to-noise ratio),
            and optional 'conditions'. Default is True.
        **kwargs
            Additional keyword arguments passed to the parent ``InferenceNetwork`` initializer
            (e.g., ``name``, ``dtype``, or ``trainable``).
        """
        super().__init__(base_distribution="normal", **kwargs)

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = StableConsistencyModel.MLP_DEFAULT_CONFIG | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self.subnet_projector = keras.layers.Dense(
            units=None, bias_initializer="zeros", kernel_initializer="zeros", name="subnet_projector"
        )
        self._concatenate_subnet_input = kwargs.get("concatenate_subnet_input", True)

        weight_mlp_kwargs = weight_mlp_kwargs or {}
        weight_mlp_kwargs = StableConsistencyModel.WEIGHT_MLP_DEFAULT_CONFIG | weight_mlp_kwargs
        self.weight_fn = MLP(**weight_mlp_kwargs)

        self.weight_fn_projector = keras.layers.Dense(
            units=1, bias_initializer="zeros", kernel_initializer="zeros", name="weight_fn_projector"
        )

        embedding_kwargs = embedding_kwargs or {}
        self.time_emb = FourierEmbedding(**embedding_kwargs)
        self.time_emb_dim = self.time_emb.embed_dim

        self.sigma = sigma
        self.seed_generator = keras.random.SeedGenerator()

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "sigma": self.sigma,
            "time_emb": self.time_emb,
            "concatenate_subnet_input": self._concatenate_subnet_input,
        }

        return base_config | serialize(config)

    @staticmethod
    def _discretize_time(num_steps: int, rho: float = 3.5):
        t = keras.ops.linspace(0.0, pi / 2, num_steps)
        times = keras.ops.exp((t - pi / 2) * rho) * pi / 2
        times = keras.ops.concatenate([keras.ops.zeros((1,)), times[1:]], axis=0)

        # if rho is set too low, bad schedules can occur
        if times[1] > StableConsistencyModel.EPS_WARN:
            logging.warning("Warning: The last time step is large.")
            logging.warning(f"Increasing rho (was {rho}) or n_steps (was {num_steps}) might improve results.")
        return times

    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)
        self.subnet_projector.units = xz_shape[-1]

        # construct input shape for subnet and subnet projector
        input_shape = list(xz_shape)

        if self._concatenate_subnet_input:
            # construct time vector
            input_shape[-1] += self.time_emb_dim + 1
            if conditions_shape is not None:
                input_shape[-1] += conditions_shape[-1]
            input_shape = tuple(input_shape)

            self.subnet.build(input_shape)
            input_shape = self.subnet.compute_output_shape(input_shape)
        else:
            # Multiple separate inputs
            time_shape = tuple(xz_shape[:-1]) + (self.time_emb_dim + 1,)  # same batch/sequence dims, 1 feature
            self.subnet.build(x_shape=xz_shape, t_shape=time_shape, conditions_shape=conditions_shape)
            input_shape = self.subnet.compute_output_shape(
                x_shape=xz_shape, t_shape=time_shape, conditions_shape=conditions_shape
            )
        self.subnet_projector.build(input_shape)

        # input shape for time embedding
        self.time_emb.build((xz_shape[0], 1))

        # input shape for weight function and projector
        input_shape = (xz_shape[0], 1)
        self.weight_fn.build(input_shape)
        input_shape = self.weight_fn.compute_output_shape(input_shape)
        self.weight_fn_projector.build(input_shape)

    def _apply_subnet(
        self, x: Tensor, t: Tensor, conditions: Tensor = None, training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """
        Prepares and passes the input to the subnet either by concatenating the latent variable `x`,
        the time `t`, and optional conditions or by returning them separately.

        Parameters
        ----------
        x : Tensor
            The parameter tensor, typically of shape (..., D), but can vary.
        t : Tensor
            The time tensor, typically of shape (..., 1).
        conditions : Tensor, optional
            The optional conditioning tensor (e.g. parameters).
        training : bool, optional
            The training mode flag, which can be used to control behavior during training.

        Returns
        -------
        Tensor
            The output tensor from the subnet.
        """
        if self._concatenate_subnet_input:
            xtc = tensor_utils.concatenate_valid([x, t, conditions], axis=-1)
            return self.subnet(xtc, training=training)
        else:
            return self.subnet(x=x, t=t, conditions=conditions, training=training)

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
            Additional keyword arguments. Include `steps` (default: 15) to
            adjust the number of sampling steps.

        Returns
        -------
        x            : Tensor
            The approximate samples
        """
        steps = kwargs.get("steps", 15)
        rho = kwargs.get("rho", 3.5)

        # noise distribution has variance sigma
        x = keras.ops.copy(z) * self.sigma
        discretized_time = keras.ops.flip(self._discretize_time(steps, rho=rho), axis=-1)
        t = keras.ops.full((*keras.ops.shape(x)[:-1], 1), discretized_time[0], dtype=x.dtype)
        x = self.consistency_function(x, t, conditions=conditions)
        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)
            x_n = ops.cos(t) * x + ops.sin(t) * noise
            t = keras.ops.full_like(t, discretized_time[n])
            x = self.consistency_function(x_n, t, conditions=conditions)
        return x

    def consistency_function(
        self,
        x: Tensor,
        t: Tensor,
        conditions: Tensor = None,
        training: bool = False,
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
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the inner network.
        """
        subnet_out = self._apply_subnet(x / self.sigma, self.time_emb(t), conditions, training=training)
        f = self.subnet_projector(subnet_out)
        out = ops.cos(t) * x - ops.sin(t) * self.sigma * f
        return out

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        # $# Implements Algorithm 1 from [1]

        # training parameters
        p_mean = -1.0
        p_std = 1.6

        c = 0.1

        # generate noise vector
        z = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator) * self.sigma

        # sample time
        tau = (
            keras.random.normal(keras.ops.shape(x)[:1], dtype=keras.ops.dtype(x), seed=self.seed_generator) * p_std
            + p_mean
        )
        t_ = ops.arctan(ops.exp(tau) / self.sigma)
        t = expand_right_as(t_, x)

        # generate noisy sample
        xt = ops.cos(t) * x + ops.sin(t) * z

        # calculate estimator for dx_t/dt
        dxtdt = ops.cos(t) * z - ops.sin(t) * x

        r = 1.0  # TODO: if consistency distillation training (not supported yet) is unstable, add schedule here

        def f_teacher(x, t):
            o = self._apply_subnet(x, self.time_emb(t), conditions, training=stage == "training")
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
        subnet_out = self._apply_subnet(xt / self.sigma, self.time_emb(t), conditions, training=stage == "training")
        student_out = self.subnet_projector(subnet_out)

        # calculate the tangent
        g = -(ops.cos(t) ** 2) * (self.sigma * teacher_output - dxtdt) - r * ops.cos(t) * ops.sin(t) * (
            xt + self.sigma * cos_sin_dFdt
        )

        # apply normalization to stabilize training
        g = g / (ops.norm(g, axis=-1, keepdims=True) + c)

        # compute adaptive weights and calculate loss
        w = self.weight_fn_projector(self.weight_fn(expand_right_to(t_, 2)))

        D = ops.shape(x)[-1]

        loss = ops.mean(
            (ops.exp(w) / D)
            * ops.mean(
                ops.reshape(((student_out - teacher_output - g) ** 2), (ops.shape(teacher_output)[0], -1)), axis=-1
            )
            - w
        )

        return base_metrics | {"loss": loss}
