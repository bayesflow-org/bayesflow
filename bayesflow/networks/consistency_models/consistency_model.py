import math

import keras
from keras import ops
from keras.saving import (
    register_keras_serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import find_network, keras_kwargs


from ..inference_network import InferenceNetwork


@register_keras_serializable(package="bayesflow.networks")
class ConsistencyModel(InferenceNetwork):
    """Implements a Consistency Model with Consistency Training (CT) as
    described in [1-2]. The adaptations to CT described in [2] were taken
    into account in this implementation.

    [1] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023).
    Consistency Models.
    arXiv preprint arXiv:2303.01469

    [2] Song, Y., & Dhariwal, P. (2023).
    Improved Techniques for Training Consistency Models:
    arXiv preprint arXiv:2310.14189
    Discussion: https://openreview.net/forum?id=WNzy9bRDvG
    """

    def __init__(
        self,
        total_steps: int | float,
        subnet: str | type = "mlp",
        max_time: int | float = 200,
        sigma2: float = 1.0,
        eps: float = 0.001,
        s0: int | float = 10,
        s1: int | float = 50,
        **kwargs,
    ):
        """Creates an instance of a consistency model (CM) to be used
        for standalone consistency training (CT).

        Parameters:
        -----------
        total_steps : int
            The total number of training steps, can be calculate as
            number of epochs * number of batches
        subnet      : str or type, optional, default: "mlp"
            A neural network type for the consistency model, will be
            instantiated using subnet_kwargs.
        max_time : int or float, optional, default: 200.0
            The maximum time of the diffusion
        sigma2      : float or Tensor of dimension (input_dim, 1),
                      optional, default: 1.0
            Controls the shape of the skip-function
        eps         : float, optional, default: 0.001
            The minimum time
        s0          : int or float, optional, default: 10
            Initial number of discretization steps
        s1          : int or float, optional, default: 50
            Final number of discretization steps
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments
        """
        # Normal is the only supported base distribution for CMs
        super().__init__(base_distribution="normal", **keras_kwargs(kwargs))

        self.total_steps = float(total_steps)

        self.student = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.student_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")

        self.sigma2 = ops.convert_to_tensor(sigma2)
        self.sigma = ops.sqrt(sigma2)
        self.eps = eps
        self.max_time = max_time
        self.c_huber = None
        self.c_huber2 = None

        self.s0 = float(s0)
        self.s1 = float(s1)
        self.current_step = 0.0

        self.seed_generator = keras.random.SeedGenerator()

    def _schedule_discretization(self) -> int:
        """Schedule function for adjusting the discretization level `N` during
        the course of training.

        Implements the function N(k) from [2], Section 3.4.
        """

        k_ = math.floor(self.total_steps / (math.log(self.s1 / self.s0) / math.log(2.0) + 1.0))
        out = min(self.s0 * math.pow(2.0, math.floor(self.current_step / k_)), self.s1) + 1.0
        return int(out)

    def _discretize_time(self, num_steps, rho=7.0):
        """Function for obtaining the discretized time according to [2],
        Section 2, bottom of page 2.
        """

        N = num_steps + 1.0
        indices = ops.arange(1, N + 1, dtype="float32")
        one_over_rho = 1.0 / rho
        discretized_time = (
            self.eps**one_over_rho
            + (indices - 1.0) / (N - 1.0) * (self.max_time**one_over_rho - self.eps**one_over_rho)
        ) ** rho
        return discretized_time

    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)
        self.student_projector.units = xz_shape[-1]

        input_shape = list(xz_shape)

        # time vector
        input_shape[-1] += 1

        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.student.build(input_shape)

        input_shape = self.student.compute_output_shape(input_shape)
        self.student_projector.build(input_shape)

        # Choose coefficient according to [2] Section 3.3
        self.c_huber = 0.00054 * math.sqrt(xz_shape[-1])
        self.c_huber2 = self.c_huber**2

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

    def _forward_train(self, x: Tensor, noise: Tensor, t: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        """Forward function for training. Calls consistency function with
        noisy input
        """
        inp = x + t * noise
        return self.consistency_function(inp, t, conditions=conditions, **kwargs)

    def _forward(self, x: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        # Consistency Models only learn the direction from noise distribution
        # to target distribution, so we cannot implement this function.
        raise NotImplementedError("Consistency Models are not invertible")

    def _inverse(self, z: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        """Generate random draws from the approximate target distribution
        using the multistep sampling algorithm from [1], Algorithm 1.

        Parameters
        ----------
        z           : Tensor
            Samples from a standard normal distribution
        conditions  : Tensor, optional, default: None
            Conditions for a approximate conditional distribution
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
        x = self.consistency_function(x, t, conditions=conditions)
        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)
            x_n = x + keras.ops.sqrt(keras.ops.square(discretized_time[n]) - self.eps**2) * noise
            t = keras.ops.full_like(t, discretized_time[n])
            x = self.consistency_function(x_n, t, conditions=conditions)
        return x

    def consistency_function(self, x: Tensor, t: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        """Compute consistency function.

        Parameters
        ----------
        x           : Tensor
            Input vector
        t           : Tensor
            Vector of time samples in [eps, T]
        conditions  : Tensor
            The conditioning vector
        **kwargs    : dict, optional, default: {}
            Additional keyword arguments passed to the network.
        """

        if conditions is not None:
            xtc = ops.concatenate([x, t, conditions], axis=-1)
        else:
            xtc = ops.concatenate([x, t], axis=-1)

        f = self.student_projector(self.student(xtc, **kwargs))

        # Compute skip and out parts (vectorized, since self.sigma2 is of shape (1, input_dim)
        # Thus, we can do a cross product with the time vector which is (batch_size, 1) for
        # a resulting shape of cskip and cout of (batch_size, input_dim)
        skip = self.sigma2 / ((t - self.eps) ** 2 + self.sigma2)
        out = self.sigma * (t - self.eps) / (ops.sqrt(self.sigma2 + t**2))

        out = skip * x + out * f
        return out

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        # The discretization schedule requires the number of passed training steps.
        # To be independent of external information, we track it here.
        self.current_step += 1

        current_num_steps = self._schedule_discretization()
        discretized_time = self._discretize_time(current_num_steps)

        # Randomly sample t_n and t_[n+1] and reshape to (batch_size, 1)
        # adapted noise schedule from [2], Section 3.5
        p_mean = -1.1
        p_std = 2.0
        log_p = ops.log(
            ops.erf((ops.log(discretized_time[1:]) - p_mean) / (ops.sqrt(2.0) * p_std))
            - ops.erf((ops.log(discretized_time[:-1]) - p_mean) / (ops.sqrt(2.0) * p_std))
        )
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
        loss = ops.mean(lam * (ops.sqrt(ops.square(teacher_out - student_out) + self.c_huber2) - self.c_huber))

        return base_metrics | {"loss": loss}
