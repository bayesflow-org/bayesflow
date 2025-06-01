import keras
from keras import ops

import numpy as np

from bayesflow.types import Tensor
from bayesflow.utils import find_network, layer_kwargs, weighted_mean
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..inference_network import InferenceNetwork


@serializable("bayesflow.networks")
class ConsistencyModel(InferenceNetwork):
    """(IN) Implements a Consistency Model with Consistency Training (CT) as described in [1-2].
    The adaptations to CT described in [2] were taken into account in our implementation for ABI [3].

    [1] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023). Consistency Models. arXiv preprint arXiv:2303.01469

    [2] Song, Y., & Dhariwal, P. (2023). Improved Techniques for Training Consistency Models.
    arXiv preprint arXiv:2310.14189. Discussion: https://openreview.net/forum?id=WNzy9bRDvG

    [3] Schmitt, M., Pratz, V., Köthe, U., Bürkner, P. C., & Radev, S. T. (2023). Consistency models for scalable and
    fast simulation-based inference. arXiv preprint arXiv:2312.05440.
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    def __init__(
        self,
        total_steps: int | float,
        subnet: str | keras.Layer = "mlp",
        max_time: int | float = 200,
        sigma2: float = 1.0,
        eps: float = 0.001,
        s0: int | float = 10,
        s1: int | float = 50,
        subnet_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        """Creates an instance of a consistency model (CM) to be used for standalone consistency training (CT).

        Parameters
        ----------
        total_steps : int
            The total number of training steps, must be calculated as number of epochs * number of batches
            and cannot be inferred during construction time.
        subnet      : str or type, optional, default: "mlp"
            A neural network type for the consistency model, will be
            instantiated using subnet_kwargs.
        max_time : int or float, optional, default: 200.0
            The maximum time of the diffusion
        sigma2      : float or Tensor of dimension (input_dim, 1), optional, default: 1.0
            Controls the shape of the skip-function
        eps         : float, optional, default: 0.001
            The minimum time
        s0          : int or float, optional, default: 10
            Initial number of discretization steps
        s1          : int or float, optional, default: 50
            Final number of discretization steps
        subnet_kwargs: dict[str, any], optional
            Keyword arguments passed to the subnet constructor or used to update the default MLP settings.
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments
        """
        super().__init__(base_distribution="normal", **kwargs)

        self.total_steps = float(total_steps)

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = ConsistencyModel.MLP_DEFAULT_CONFIG | subnet_kwargs

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.output_projector = keras.layers.Dense(
            units=None, bias_initializer="zeros", kernel_initializer="zeros", name="output_projector"
        )

        self.sigma2 = ops.convert_to_tensor(sigma2)
        self.sigma = ops.sqrt(sigma2)
        self.eps = eps
        self.max_time = max_time
        self.c_huber = None
        self.c_huber2 = None

        self.s0 = float(s0)
        self.s1 = float(s1)

        # create variable that works with JIT compilation
        self.current_step = self.add_weight(name="current_step", initializer="zeros", trainable=False, dtype="int")
        self.current_step.assign(0)

        self.seed_generator = keras.random.SeedGenerator()

    @property
    def student(self):
        return self.subnet

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "total_steps": self.total_steps,
            "subnet": self.subnet,
            "max_time": self.max_time,
            "sigma2": self.sigma2,
            "eps": self.eps,
            "s0": self.s0,
            "s1": self.s1,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)

    def _schedule_discretization(self, step) -> float:
        """Schedule function for adjusting the discretization level `N` during
        the course of training.

        Implements the function N(k) from [2], Section 3.4.
        """

        k_ = ops.floor(self.total_steps / (ops.log(self.s1 / self.s0) / ops.log(2.0) + 1.0))
        out = ops.minimum(self.s0 * ops.power(2.0, ops.floor(step / k_)), self.s1) + 1.0
        return out

    def _discretize_time(self, num_steps, rho=7.0):
        """Function for obtaining the discretized time according to [2],
        Section 2, bottom of page 2.
        """

        N = num_steps + 1
        indices = ops.arange(1, N + 1, dtype="float32")
        one_over_rho = 1.0 / rho
        discretized_time = (
            self.eps**one_over_rho
            + (indices - 1.0) / (ops.cast(N, "float32") - 1.0) * (self.max_time**one_over_rho - self.eps**one_over_rho)
        ) ** rho
        return discretized_time

    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)

        self.output_projector.units = xz_shape[-1]

        input_shape = list(xz_shape)

        # time vector
        input_shape[-1] += 1

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.subnet.build(input_shape)

        input_shape = self.subnet.compute_output_shape(input_shape)
        self.output_projector.build(input_shape)

        # Choose coefficient according to [2] Section 3.3
        self.c_huber = 0.00054 * ops.sqrt(xz_shape[-1])
        self.c_huber2 = self.c_huber**2

        # Calculate discretization schedule in advance
        # The Jax compiler requires fixed-size arrays, so we have
        # to store all the discretized_times in one matrix in advance
        # and later only access the relevant entries.

        # First, we calculate all unique numbers of discretization steps n
        # in a loop, as self.total_steps might be large
        self.max_n = int(self._schedule_discretization(self.total_steps))

        if self.max_n != self.s1 + 1:
            raise ValueError("The maximum number of discretization steps must be equal to s1 + 1.")

        unique_n = set()
        for step in range(int(self.total_steps)):
            unique_n.add(int(self._schedule_discretization(step)))
        unique_n = sorted(list(unique_n))

        # Next, we calculate the discretized times for each n
        # and establish a mapping between n and the position i of the
        # discretizated times in the vector
        discretized_times = np.zeros((len(unique_n), self.max_n + 1))
        discretization_map = np.zeros((self.max_n + 1,), dtype=np.int32)
        for i, n in enumerate(unique_n):
            disc = ops.convert_to_numpy(self._discretize_time(n))
            discretized_times[i, : len(disc)] = disc
            discretization_map[n] = i

        # Finally, we convert the vectors to tensors
        self.discretized_times = ops.convert_to_tensor(discretized_times, dtype="float32")
        self.discretization_map = ops.convert_to_tensor(discretization_map)

    def _forward_train(
        self, x: Tensor, noise: Tensor, t: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> Tensor:
        """Forward function for training. Calls consistency function with noisy input"""
        inp = x + t * noise
        return self.consistency_function(inp, t, conditions=conditions, training=training)

    def _forward(self, x: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        # Consistency Models only learn the direction from noise distribution
        # to target distribution, so we cannot implement this function.
        raise NotImplementedError("Consistency Models are not invertible")

    def _inverse(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> Tensor:
        """Generate random draws from the approximate target distribution
        using the multistep sampling algorithm from [1], Algorithm 1.

        Parameters
        ----------
        z           : Tensor
            Samples from a standard normal distribution
        conditions  : Tensor, optional, default: None
            Conditions for the approximate conditional distribution
        training    : bool, optional, default: True
            Whether internal layers (e.g., dropout) should behave in train or inference mode.
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments. Include `steps` (default: 10) to
            adjust the number of sampling steps.

        Returns
        -------
        x            : Tensor
            The approximate samples
        """
        steps = kwargs.get("steps", 10)
        x = keras.ops.copy(z) * self.max_time
        discretized_time = keras.ops.flip(self._discretize_time(steps), axis=-1)
        t = keras.ops.full((*keras.ops.shape(x)[:-1], 1), discretized_time[0], dtype=x.dtype)

        x = self.consistency_function(x, t, conditions=conditions, training=training)

        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)
            x_n = x + keras.ops.sqrt(keras.ops.square(discretized_time[n]) - self.eps**2) * noise
            t = keras.ops.full_like(t, discretized_time[n])
            x = self.consistency_function(x_n, t, conditions=conditions, training=training)
        return x

    def consistency_function(self, x: Tensor, t: Tensor, conditions: Tensor = None, training: bool = False) -> Tensor:
        """Compute consistency function.

        Parameters
        ----------
        x           : Tensor
            Input vector
        t           : Tensor
            Vector of time samples in [eps, T]
        conditions  : Tensor
            The conditioning vector
        training    : bool, optional, default: True
            Whether internal layers (e.g., dropout) should behave in train or inference mode.
        """

        if conditions is not None:
            xtc = ops.concatenate([x, t, conditions], axis=-1)
        else:
            xtc = ops.concatenate([x, t], axis=-1)

        f = self.output_projector(self.subnet(xtc, training=training))

        # Compute skip and out parts (vectorized, since self.sigma2 is of shape (1, input_dim)
        # Thus, we can do a cross product with the time vector which is (batch_size, 1) for
        # a resulting shape of cskip and cout of (batch_size, input_dim)
        skip = self.sigma2 / ((t - self.eps) ** 2 + self.sigma2)
        out = self.sigma * (t - self.eps) / (ops.sqrt(self.sigma2 + t**2))

        out = skip * x + out * f
        return out

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        # The discretization schedule requires the number of passed training steps.
        # To be independent of external information, we track it here.
        if stage == "training":
            self.current_step.assign_add(1)
            self.current_step.assign(ops.minimum(self.current_step, self.total_steps - 1))

        discretization_index = ops.take(
            self.discretization_map, ops.cast(self._schedule_discretization(self.current_step), "int")
        )
        discretized_time = ops.take(self.discretized_times, discretization_index, axis=0)

        # Randomly sample t_n and t_[n+1] and reshape to (batch_size, 1)
        # adapted noise schedule from [2], Section 3.5
        p_mean = -1.1
        p_std = 2.0
        p = ops.where(
            discretized_time[1:] > 0.0,
            ops.erf((ops.log(discretized_time[1:]) - p_mean) / (ops.sqrt(2.0) * p_std))
            - ops.erf((ops.log(discretized_time[:-1]) - p_mean) / (ops.sqrt(2.0) * p_std)),
            0.0,
        )

        log_p = ops.log(p)
        times = keras.random.categorical(ops.expand_dims(log_p, 0), ops.shape(x)[0], seed=self.seed_generator)[0]
        t1 = ops.take(discretized_time, times)[..., None]
        t2 = ops.take(discretized_time, times + 1)[..., None]

        # generate noise vector
        noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)

        teacher_out = self._forward_train(x, noise, t1, conditions=conditions, training=stage == "training")
        # difference between teacher and student: different time,
        # and no gradient for the teacher
        teacher_out = ops.stop_gradient(teacher_out)
        student_out = self._forward_train(x, noise, t2, conditions=conditions, training=stage == "training")

        # weighting function, see [2], Section 3.1
        lam = 1 / (t2 - t1)

        # Pseudo-huber loss, see [2], Section 3.3
        loss = lam * (ops.sqrt(ops.square(teacher_out - student_out) + self.c_huber2) - self.c_huber)
        loss = weighted_mean(loss, sample_weight)

        return base_metrics | {"loss": loss}
