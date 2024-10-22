from collections.abc import Sequence
import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape, Tensor
from bayesflow.utils import expand_right_as, keras_kwargs, optimal_transport
from ..inference_network import InferenceNetwork
from .integrators import EulerIntegrator
from .integrators import RK2Integrator
from .integrators import RK4Integrator


@serializable(package="bayesflow.networks")
class FlowMatching(InferenceNetwork):
    """Implements Optimal Transport Flow Matching, originally introduced as Rectified Flow,
    with ideas incorporated from [1-3].

    [1] Rectified Flow: arXiv:2209.03003
    [2] Flow Matching: arXiv:2210.02747
    [3] Optimal Transport Flow Matching: arXiv:2302.00482
    """

    def __init__(
        self,
        subnet: str | type = "mlp",
        base_distribution: str = "normal",
        integrator: str = "euler",
        use_optimal_transport: bool = False,
        optimal_transport_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))

        self.use_optimal_transport = use_optimal_transport
        self.optimal_transport_kwargs = optimal_transport_kwargs or {
            "method": "sinkhorn",
            "cost": "euclidean",
            "regularization": 0.1,
            "max_steps": 1000,
            "tolerance": 1e-4,
        }

        self.seed_generator = keras.random.SeedGenerator()

        match integrator:
            case "euler":
                self.integrator = EulerIntegrator(subnet, **kwargs)
            case "rk2":
                self.integrator = RK2Integrator(subnet, **kwargs)
            case "rk4":
                self.integrator = RK4Integrator(subnet, **kwargs)
            case _:
                raise NotImplementedError(f"No support for {integrator} integration")

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        super().build(xz_shape)
        self.integrator.build(xz_shape, conditions_shape)

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

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 100)

        if density:
            z, trace = self.integrator(x, conditions=conditions, steps=steps, density=True)
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob + trace
            return z, log_density

        z = self.integrator(x, conditions=conditions, steps=steps, density=False)
        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        steps = kwargs.get("steps", 100)

        if density:
            x, trace = self.integrator(z, conditions=conditions, steps=steps, density=True, inverse=True)
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob - trace
            return x, log_density

        x = self.integrator(z, conditions=conditions, steps=steps, density=False, inverse=True)
        return x

    def compute_metrics(
        self, x: Tensor | Sequence[Tensor, ...], conditions: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        if isinstance(x, Sequence):
            # already pre-configured
            x0, x1, t, x, target_velocity = x
        else:
            # not pre-configured, resample
            x1 = x
            x0 = keras.random.normal(keras.ops.shape(x1), dtype=keras.ops.dtype(x1), seed=self.seed_generator)

            if self.use_optimal_transport:
                x1, x0, conditions = optimal_transport(
                    x1, x0, conditions, seed=self.seed_generator, **self.optimal_transport_kwargs
                )

            t = keras.random.uniform((keras.ops.shape(x0)[0],), seed=self.seed_generator)
            t = expand_right_as(t, x0)

            x = t * x1 + (1 - t) * x0
            target_velocity = x1 - x0

        base_metrics = super().compute_metrics(x1, conditions, stage)

        predicted_velocity = self.integrator.velocity(x, t, conditions)

        loss = keras.losses.mean_squared_error(target_velocity, predicted_velocity)
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
