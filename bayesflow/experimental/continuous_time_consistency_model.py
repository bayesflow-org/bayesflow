import keras
from keras import ops

import numpy as np

from bayesflow.networks import MLP
from bayesflow.types import Tensor
from bayesflow.utils import (
    jvp,
    concatenate_valid,
    find_network,
    expand_right_as,
    expand_right_to,
    model_kwargs,
)
from bayesflow.utils.serialization import deserialize, serializable, serialize


from bayesflow.networks import InferenceNetwork
from bayesflow.networks.embeddings import FourierEmbedding


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class ContinuousTimeConsistencyModel(InferenceNetwork):
    """(IN) Implements an sCM (simple, stable, and scalable Consistency Model)
    with continous-time Consistency Training (CT) as described in [1].
    The sampling procedure is taken from [2].

    [1] Lu, C., & Song, Y. (2024).
    Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models
    arXiv preprint arXiv:2410.11081

    [2] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023).
    Consistency Models.
    arXiv preprint arXiv:2303.01469
    """

    def __init__(
        self,
        subnet: str | keras.Layer = "mlp",
        sigma_data: float = 1.0,
        subnet_kwargs: dict[str, any] = None,
        embedding_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        """Creates an instance of an sCM to be used for consistency training (CT).

        Parameters
        ----------
        subnet        : str or type, optional, default: "mlp"
            A neural network type for the consistency model, will be
            instantiated using subnet_kwargs.
        sigma_data    : float, optional, default: 1.0
            Standard deviation of the target distribution
        **kwargs
            Additional keyword arguments to the layer.
        """
        super().__init__(base_distribution="normal", **kwargs)

        subnet_kwargs = subnet_kwargs or {}

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.subnet_projector = keras.layers.Dense(
            units=None, bias_initializer="zeros", kernel_initializer="zeros", name="subnet_projector"
        )

        self.weight_fn = MLP([256], dropout=0.0)
        self.weight_fn_projector = keras.layers.Dense(
            units=1, bias_initializer="zeros", kernel_initializer="zeros", name="weight_fn_projector"
        )

        embedding_kwargs = embedding_kwargs or {}
        self.time_emb = FourierEmbedding(**embedding_kwargs)
        self.time_emb_dim = self.time_emb.embed_dim

        self.sigma_data = sigma_data

        self.seed_generator = keras.random.SeedGenerator()

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = model_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "sigma_data": self.sigma_data,
        }

        return base_config | serialize(config)

    def _discretize_time(self, num_steps: int, rho: float = 3.5, **kwargs):
        t = np.linspace(0.0, np.pi / 2, num_steps)
        times = np.exp((t - np.pi / 2) * rho) * np.pi / 2
        times[0] = 0.0

        # if rho is set too low, bad schedules can occur
        EPS_WARN = 0.1
        if times[1] > EPS_WARN:
            print("Warning: The last time step is large.")
            print(f"Increasing rho (was {rho}) or n_steps (was {num_steps}) might improve results.")
        return ops.convert_to_tensor(times)

    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)
        self.subnet_projector.units = xz_shape[-1]

        # construct input shape for subnet and subnet projector
        input_shape = list(xz_shape)

        # time vector
        input_shape[-1] += self.time_emb_dim + 1

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.subnet.build(input_shape)

        input_shape = self.subnet.compute_output_shape(input_shape)
        self.subnet_projector.build(input_shape)

        # input shape for time embedding
        self.time_emb.build((xz_shape[0], 1))

        # input shape for weight function and projector
        input_shape = (xz_shape[0], 1)
        self.weight_fn.build(input_shape)
        input_shape = self.weight_fn.compute_output_shape(input_shape)
        self.weight_fn_projector.build(input_shape)

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False,
        **kwargs,
    ):
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

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
            Conditions for a approximate conditional distribution
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments. Include `steps` (default: 30) to
            adjust the number of sampling steps.

        Returns
        -------
        x            : Tensor
            The approximate samples
        """
        steps = kwargs.get("steps", 15)
        rho = kwargs.get("rho", 3.5)

        # noise distribution has variance sigma_data
        x = keras.ops.copy(z) * self.sigma_data
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
        **kwargs,
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
        xtc = concatenate_valid([x / self.sigma_data, self.time_emb(t), conditions], axis=-1)
        f = self.subnet_projector(self.subnet(xtc, training=training, **kwargs))
        out = ops.cos(t) * x - ops.sin(t) * self.sigma_data * f
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
        z = (
            keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)
            * self.sigma_data
        )

        # sample time
        tau = (
            keras.random.normal(keras.ops.shape(x)[:1], dtype=keras.ops.dtype(x), seed=self.seed_generator) * p_std
            + p_mean
        )
        t_ = ops.arctan(ops.exp(tau) / self.sigma_data)
        t = expand_right_as(t_, x)

        # generate noisy sample
        xt = ops.cos(t) * x + ops.sin(t) * z

        # calculate estimator for dx_t/dt
        dxtdt = ops.cos(t) * z - ops.sin(t) * x

        r = 1.0  # TODO: if consistency distillation training (not supported yet) is unstable, add schedule here

        def f_teacher(x, t):
            o = self.subnet(concatenate_valid([x, self.time_emb(t), conditions], axis=-1), training=stage == "training")
            return self.subnet_projector(o)

        primals = (xt / self.sigma_data, t)
        tangents = (
            ops.cos(t) * ops.sin(t) * dxtdt,
            ops.cos(t) * ops.sin(t) * self.sigma_data,
        )

        teacher_output, cos_sin_dFdt = jvp(f_teacher, primals, tangents, return_output=True)
        teacher_output = ops.stop_gradient(teacher_output)
        cos_sin_dFdt = ops.stop_gradient(cos_sin_dFdt)

        # calculate output of the network
        xtc = concatenate_valid([xt / self.sigma_data, self.time_emb(t), conditions], axis=-1)
        student_out = self.subnet_projector(self.subnet(xtc, training=stage == "training"))

        # calculate the tangent
        g = -(ops.cos(t) ** 2) * (self.sigma_data * teacher_output - dxtdt) - r * ops.cos(t) * ops.sin(t) * (
            xt + self.sigma_data * cos_sin_dFdt
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
