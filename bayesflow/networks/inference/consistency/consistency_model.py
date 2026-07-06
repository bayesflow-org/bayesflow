import numpy as np

import keras
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import (
    expand_right_as,
    find_network,
    filter_kwargs,
    layer_kwargs,
    logging,
    maybe_mask_tensor,
    resolve_seed,
    sample_input_masks,
    weighted_mean,
)
from bayesflow.utils.serialization import serializable, serialize

from ...inference import InferenceNetwork
from ...defaults import TIME_MLP_DEFAULTS, DIFFUSION_TRANSFORMER_DEFAULTS


@serializable("bayesflow.networks")
class ConsistencyModel(InferenceNetwork):
    """Consistency model with consistency training (CT) for simulation-based inference.

    Implements a Consistency Model as described in [1-2], with the adaptations to
    CT from [2] incorporated for amortised Bayesian inference [3].

    Parameters
    ----------
    total_steps : int or float
        The total number of training steps, must be calculated as
        ``num_epochs * num_batches`` and cannot be inferred during construction.
    subnet : str or keras.Layer, optional
        A neural network type for the consistency model, will be instantiated using
        *subnet_kwargs*.  If a string is provided, it should be a registered name
        (e.g., ``"time_mlp"``).  If a type or ``keras.Layer`` is provided, it will
        be directly instantiated with the given *subnet_kwargs*.  Any subnet must
        accept a tuple of tensors ``(target, time, conditions)``.
        Default is ``"time_mlp"``.
    max_time : int or float, optional
        The maximum time of the diffusion, equivalent to the maximum noise level
        (``x_1 = z * max_time``).  Default is 80.
    sigma2 : float, optional
        Controls the shape of the skip-function.  Default is 1.0.
    eps : float, optional
        The minimum time.  Default is 0.001.
    s0 : int or float, optional
        Initial number of discretisation steps and s0+1 steps will be used for sampling.
         Default is 10. During sampling, you can pass `steps` to change nmuber of multistep sampling.
    s1 : int or float, optional
        Final number of discretisation steps.  Default is 150.
    rho : float, optional
        Exponent controlling the curvature of the (Karras-style) time
        discretisation schedule. Default is ``7.0``.
    noise_dist_mean : float, optional
        Mean of the log-normal proposal distribution over noise levels used to
        weight the sampled discretisation times (``P_mean`` in [2]).
        Default is ``-1.1``.
    noise_dist_std : float, optional
        Standard deviation of that proposal distribution (``P_std`` in [2]).
        Default is ``2.0``.
    subnet_kwargs : dict[str, any], optional
        Keyword arguments passed to the subnet constructor or used to update the
        default MLP settings.
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``.

    References
    ----------
    [1] Song, Y., Dhariwal, P., Chen, M. & Sutskever, I. (2023). Consistency
        Models. arXiv:2303.01469.
    [2] Song, Y., & Dhariwal, P. (2023). Improved Techniques for Training
        Consistency Models. arXiv:2310.14189.
    [3] Schmitt, M., Pratz, V., Köthe, U., Bürkner, P. C., & Radev, S. T.
        (2023). Consistency models for scalable and fast simulation-based
        inference. arXiv:2312.05440.
    """

    _SUBNET_MASK_KEYS = {
        "attention_mask",
        "target_inference_mask",
        "target_condition_mask",
        "condition_mask",
    }

    def __init__(
        self,
        total_steps: int | float,
        subnet: str | keras.Layer = "time_mlp",
        max_time: int | float = 80,
        sigma2: float = 1.0,
        eps: float = 0.001,
        s0: int | float = 10,
        s1: int | float = 150,
        rho: float = 7.0,
        noise_dist_mean: float = -1.1,
        noise_dist_std: float = 2.0,
        subnet_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(base_distribution="normal", **kwargs)

        self.total_steps = float(total_steps)

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "time_mlp":
            subnet_kwargs = TIME_MLP_DEFAULTS | subnet_kwargs
        if subnet == "diffusion_transformer":
            subnet_kwargs = DIFFUSION_TRANSFORMER_DEFAULTS | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)
        self._subnet_mask_keys = set(filter_kwargs({k: None for k in self._SUBNET_MASK_KEYS}, self.subnet.call).keys())

        self.output_projector = None

        self.sigma2 = ops.convert_to_tensor(sigma2)
        self.sigma = ops.sqrt(sigma2)
        self.eps = eps
        self.max_time = max_time
        self.rho = rho
        self.noise_dist_mean = noise_dist_mean
        self.noise_dist_std = noise_dist_std

        self.s0 = s0
        self.s1 = s1

        if self.total_steps < self.s0:
            raise ValueError(f"total_steps={self.total_steps} must be greater than or equal to s0={self.s0}.")

        # create variable that works with JIT compilation
        self._current_step = self.add_weight(name="current_step", initializer="zeros", trainable=False, dtype="int")
        self._current_step.assign(0)

        self.seed_generator = keras.random.SeedGenerator()
        self._discretized_times = None
        self._discretization_map = None
        self._c_huber = None
        self._c_huber2 = None
        self._unique_n = None
        self.drop_target_prob = kwargs.get("drop_target_prob", 0.0)
        self.drop_missing_prob = kwargs.get("drop_missing_prob", 0.0)

    @property
    def student(self):
        return self.subnet

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
            "rho": self.rho,
            "noise_dist_mean": self.noise_dist_mean,
            "noise_dist_std": self.noise_dist_std,
            "drop_target_prob": self.drop_target_prob,
            "drop_missing_prob": self.drop_missing_prob,
            # we do not need to store subnet_kwargs
        }

        return base_config | serialize(config)

    def _schedule_discretization(self, step) -> float:
        """Schedule function for adjusting the discretization level `N(k)` during
        the course of training.

        Implements the function N(k) from [2], Section 3.4.
        """
        k_ = ops.floor(self.total_steps / (ops.log(ops.floor(self.s1 / self.s0)) / ops.log(2.0) + 1.0))
        out = ops.minimum(self.s0 * ops.power(2.0, ops.floor(step / k_)), self.s1) + 1.0
        return out

    def _discretize_time(self, n_k: int) -> Tensor:
        """Function for obtaining the discretized time according to [2],
        Section 2, bottom of page 2.
        """
        indices = ops.arange(1, n_k + 1, dtype="float32")
        one_over_rho = 1.0 / self.rho
        discretized_time = (
            self.eps**one_over_rho
            + (indices - 1.0)
            / (ops.cast(n_k, "float32") - 1.0)
            * (self.max_time**one_over_rho - self.eps**one_over_rho)
        ) ** self.rho
        return discretized_time

    def build(self, xz_shape, conditions_shape=None):
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
        time_shape = (xz_shape[0], 1)  # same batch dims, 1 feature
        self.subnet.build((xz_shape, time_shape, conditions_shape))
        out_shape = self.subnet.compute_output_shape((xz_shape, time_shape, conditions_shape))
        self.output_projector.build(out_shape)

        # Choose coefficient according to [2] Section 3.3
        self._c_huber = 0.00054 * ops.sqrt(xz_shape[-1])
        self._c_huber2 = self._c_huber**2

        # Calculate discretization schedule in advance
        # The Jax compiler requires fixed-size arrays, so we have
        # to store all the discretized_times in one matrix in advance
        # and later only access the relevant entries.

        # First, we calculate all unique numbers of discretization steps n
        # in a loop, as self.total_steps might be large
        max_n = int(self._schedule_discretization(self.total_steps))
        if max_n != self.s1 + 1:
            raise ValueError("The maximum number of discretization steps must be equal to s1 + 1.")

        unique_n = set()
        for step in range(int(self.total_steps)):
            unique_n.add(int(self._schedule_discretization(step)))
        self._unique_n = sorted(list(unique_n))

        # Next, we calculate the discretized times for each n
        # and establish a mapping between n and the position i of the
        # discretized times in the vector
        discretized_times = np.zeros((len(unique_n), max_n + 1))
        discretization_map = np.zeros((max_n + 1,), dtype=np.int32)
        for i, n in enumerate(unique_n):
            disc = ops.convert_to_numpy(self._discretize_time(n))
            discretized_times[i, : len(disc)] = disc
            discretization_map[n] = i

        # Finally, we convert the vectors to tensors
        self._discretized_times = ops.convert_to_tensor(discretized_times, dtype="float32")
        self._discretization_map = ops.convert_to_tensor(discretization_map)

    def _forward_train(
        self,
        x: Tensor,
        noise: Tensor,
        t: Tensor,
        conditions: Tensor = None,
        training: bool = False,
        mask_x: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward function for training. Calls consistency function with noisy input"""
        inp = x + t * noise
        inp = maybe_mask_tensor(inp, mask=mask_x, replacement=x)
        return self.consistency_function(inp, t, conditions=conditions, training=training, **kwargs)

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
            Additional keyword arguments. Include `steps` (default: s0+1) to
            adjust the number of sampling steps. Subnet-related kwargs (e.g., masks)
            are passed to the subnet.

        Returns
        -------
        x            : Tensor
            The approximate samples
        """
        seed = resolve_seed(kwargs.pop("seed", None)) or self.seed_generator
        # Extract subnet masks from kwargs
        subnet_kwargs = self._collect_mask_kwargs(self._subnet_mask_keys, kwargs)
        steps = int(kwargs.get("steps", self.s0 + 1))

        if steps not in self._unique_n:
            logging.warning(
                "The number of discretization steps is not equal to the number of unique steps used during training. "
                "This might lead to suboptimal sample quality."
            )

        x = keras.ops.copy(z) * self.max_time
        discretized_time = keras.ops.flip(self._discretize_time(steps), axis=-1)
        t = keras.ops.full((*keras.ops.shape(x)[:-1], 1), discretized_time[0], dtype=x.dtype)

        # Apply user-provided target mask if available
        target_inference_mask = kwargs.get("target_inference_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_inference_mask is not None:
            target_inference_mask = keras.ops.broadcast_to(target_inference_mask, keras.ops.shape(x))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(x))
            x = maybe_mask_tensor(x, mask=target_inference_mask, replacement=targets_fixed)

        x = self.consistency_function(x, t, conditions=conditions, training=training, **subnet_kwargs)
        x = maybe_mask_tensor(x, mask=target_inference_mask, replacement=targets_fixed)

        for n in range(1, steps):
            noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=seed)
            x_n = x + keras.ops.sqrt(keras.ops.square(discretized_time[n]) - self.eps**2) * noise
            t = keras.ops.full_like(t, discretized_time[n])
            x_n = maybe_mask_tensor(x_n, mask=target_inference_mask, replacement=targets_fixed)
            x = self.consistency_function(x_n, t, conditions=conditions, training=training, **subnet_kwargs)
            x = maybe_mask_tensor(x, mask=target_inference_mask, replacement=targets_fixed)
        return x

    def consistency_function(
        self,
        x: Tensor,
        t: Tensor,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
    ) -> Tensor:
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
        **kwargs    : dict, optional
            Additional keyword arguments to pass to the subnet.
        """
        subnet_out = self.subnet((x, t / self.max_time, conditions), training=training, **kwargs)

        f = self.output_projector(subnet_out)

        # Compute skip and out parts (vectorized, since self.sigma2 is of shape (1, input_dim)
        # Thus, we can do a cross product with the time vector which is (batch_size, 1) for
        # a resulting shape of cskip and cout of (batch_size, input_dim)
        skip = self.sigma2 / ((t - self.eps) ** 2 + self.sigma2)
        out = self.sigma * (t - self.eps) / (ops.sqrt(self.sigma2 + t**2))

        out = skip * x + out * f
        return out

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        training = stage == "training"

        # The discretization schedule requires the number of passed training steps.
        # To be independent of external information, we track it here.
        if training:
            self._current_step.assign_add(1)
            self._current_step.assign(ops.minimum(self._current_step, self.total_steps - 1))

        discretization_index = ops.take(
            self._discretization_map, ops.cast(self._schedule_discretization(self._current_step), "int")
        )
        discretized_time = ops.take(self._discretized_times, discretization_index, axis=0)

        # Randomly sample t_n and t_[n+1] and reshape to (batch_size, 1)
        # adapted noise schedule from [2], Section 3.5
        p = ops.where(
            discretized_time[1:] > 0.0,
            ops.erf((ops.log(discretized_time[1:]) - self.noise_dist_mean) / (ops.sqrt(2.0) * self.noise_dist_std))
            - ops.erf((ops.log(discretized_time[:-1]) - self.noise_dist_mean) / (ops.sqrt(2.0) * self.noise_dist_std)),
            0.0,
        )

        log_p = ops.log(p)
        times = keras.random.categorical(ops.expand_dims(log_p, 0), ops.shape(x)[0], seed=self.seed_generator)[0]
        t1 = expand_right_as(ops.take(discretized_time, times), x)
        t2 = expand_right_as(ops.take(discretized_time, times + 1), x)

        # generate noise vector
        noise = keras.random.normal(keras.ops.shape(x), dtype=keras.ops.dtype(x), seed=self.seed_generator)

        # Generate target / condition / missingness masks
        subnet_kwargs = self._collect_mask_kwargs(self._subnet_mask_keys, kwargs)
        mask_x, loss_mask, subnet_kwargs = sample_input_masks(
            self.subnet,
            x,
            conditions,
            subnet_kwargs,
            training,
            self.drop_target_prob,
            self.drop_missing_prob,
            self.seed_generator,
        )

        teacher_out = self._forward_train(
            x, noise, t1, conditions=conditions, training=training, mask_x=mask_x, **subnet_kwargs
        )
        # difference between teacher and student: different time, and no gradient for the teacher
        teacher_out = ops.stop_gradient(teacher_out)
        student_out = self._forward_train(
            x, noise, t2, conditions=conditions, training=training, mask_x=mask_x, **subnet_kwargs
        )

        # weighting function, see [2], Section 3.1
        lam = 1 / (t2 - t1)

        # Pseudo-huber loss, see [2], Section 3.3
        loss = lam * (ops.sqrt(loss_mask * ops.square(teacher_out - student_out) + self._c_huber2) - self._c_huber)
        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}
