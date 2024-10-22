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
    """Implements a consistency model according to https://arxiv.org/abs/2303.01469"""

    def __init__(
        self,
        total_steps: int | float,
        subnet: str | type = "mlp",
        base_distribution: str = "normal",
        max_time: int | float = 200,
        sigma2: float = 1.0,
        eps: float = 0.001,
        s0: int | float = 10,
        s1: int | float = 50,
        **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))

        self.total_steps = float(total_steps)

        self.student = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.student_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")
        self.teacher = None
        self.teacher_projector = None

        self.sigma2 = ops.convert_to_tensor(sigma2)
        self.sigma = ops.sqrt(sigma2)
        self.eps = eps
        self.max_time = max_time
        self.c_huber = None
        self.c_huber2 = None

        self.s0 = float(s0)
        self.s1 = float(s1)
        self.current_step = 0.0

    def _schedule_discretization(self) -> int:
        """Schedule function for adjusting the discretization level `N` during the course
        of training.

        Implements the function N(k) from https://arxiv.org/abs/2310.14189, Section 3.4.
        """

        k_ = math.floor(self.total_steps / (math.log(self.s1 / self.s0) / math.log(2.0) + 1.0))
        out = min(self.s0 * math.pow(2.0, math.floor(self.current_step / k_)), self.s1) + 1.0
        return int(out)

    def discretize_time(self, num_steps, rho=7.0):
        """Function for obtaining the discretized time according to
        https://arxiv.org/pdf/2310.14189.pdf, Section 2, bottom of page 2.
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

        # Clone
        self.teacher = keras.models.clone_model(self.student)
        self.teacher_projector = keras.models.clone_model(self.student_projector)
        self.teacher.set_weights(self.student.weights)
        self.teacher_projector.set_weights(self.student_projector)
        self.teacher.trainable = False
        self.student_projector.trainable = False

        # Choose coefficient according to https://arxiv.org/pdf/2310.14189.pdf, Section 3.3
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

    def _forward(self, x: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        pass

    def _inverse(self, z: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        pass

    def consistency_function(
        self, x: Tensor, t: Tensor, conditions: Tensor = None, student: bool = True, **kwargs
    ) -> Tensor:
        """Compute consistency function with either the student or the teacher network."""

        if conditions is not None:
            xtc = ops.concatenate([x, t, conditions], axis=-1)
        else:
            xtc = ops.concatenate([x, t], axis=-1)

        # Compute either student or teacher output (no grads for teacher during training)
        if student:
            f = self.student_projector(self.student(xtc, **kwargs))
        else:
            f = self.teacher_projector(self.teacher(xtc, **kwargs))

        # Compute skip and out parts (vectorized, since self.sigma2 is of shape (1, input_dim)
        # Thus, we can do a cross product with the time vector which is (batch_size, 1) for
        # a resulting shape of cskip and cout of (batch_size, input_dim)
        skip = self.sigma2 / ((t - self.eps) ** 2 + self.sigma2)
        out = self.sigma * (t - self.eps) / (ops.sqrt(self.sigma2 + t**2))

        out = skip * x + out * f
        return out

    def update_teacher(self):
        """
        Update function for copying student network weights to teacher network weights.
        Should be called after the optimizer update of the student. EMA was dropped,
        see https://arxiv.org/pdf/2310.14189.pdf, Section 3.2.
        """

        for w_teacher, w_student in zip(self.teacher.weights, self.student.weights):
            w_teacher.assign(keras.ops.stop_gradient(w_student))

        for w_teacher, w_student in zip(self.teacher_projector.weights, self.student_projector.weights):
            w_teacher.assign(keras.ops.stop_gradient(w_student))

    def compute_metrics(self, data: dict[str, Tensor], stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(data, stage=stage)

        self.current_step += 1

        x = data["inference_variables"]
        c = data.get("inference_conditions")

        # z = self.base_distribution.sample((ops.shape(x)[0],))

        current_num_steps = self._schedule_discretization()
        discretized_time = self.discretize_time(current_num_steps)

        # Randomly sample t_n and t_[n+1] and reshape to (batch_size, 1)
        # adapted noise schedule from https://arxiv.org/pdf/2310.14189.pdf,
        # Section 3.5
        p_mean = -1.1
        p_std = 2.0
        log_p = ops.log(
            ops.erf((ops.log(discretized_time[1:]) - p_mean) / (ops.sqrt(2.0) * p_std))
            - ops.erf((ops.log(discretized_time[:-1]) - p_mean) / (ops.sqrt(2.0) * p_std))
        )
        times = keras.random.categorical([log_p], ops.shape(x)[0])[0]
        t1 = ops.take(discretized_time, times)[..., None]
        t2 = ops.take(discretized_time, times + 1)[..., None]

        teacher_out = self._forward(x, conditions=c, student=False, training=stage == "training")
        teacher_out = ops.stop_gradient(teacher_out)
        student_out = self._forward(x, conditions=c, student=True, training=stage == "training")

        # weighting function, see https://arxiv.org/pdf/2310.14189.pdf, Section 3.1
        lam = 1 / (t2 - t1)

        # Pseudo-huber loss, see https://arxiv.org/pdf/2310.14189.pdf, Section 3.3
        loss = ops.mean(lam * (ops.sqrt(ops.square(teacher_out - student_out) + self.c_huber2) - self.c_huber))

        return base_metrics | {"loss": loss}
