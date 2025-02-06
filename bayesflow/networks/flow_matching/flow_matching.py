from collections.abc import Sequence

import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape, Tensor
from bayesflow.utils import (
    expand_right_as,
    find_network,
    integrate,
    jacobian_trace,
    keras_kwargs,
    optimal_transport,
    serialize_value_or_type,
    deserialize_value_or_type,
)
from ..inference_network import InferenceNetwork


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
        use_optimal_transport: bool = False,
        optimal_transport_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))

        self.use_optimal_transport = use_optimal_transport

        if optimal_transport_kwargs is None:
            optimal_transport_kwargs = {
                "method": "sinkhorn",
                "cost": "euclidean",
                "regularization": 0.1,
                "max_steps": 1000,
                "tolerance": 1e-4,
            }

        self.optimal_transport_kwargs = optimal_transport_kwargs

        self.seed_generator = keras.random.SeedGenerator()

        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros")

        # serialization: store all parameters necessary to call __init__
        self.config = {
            "base_distribution": base_distribution,
            "use_optimal_transport": use_optimal_transport,
            "optimal_transport_kwargs": optimal_transport_kwargs,
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        super().build(xz_shape, conditions_shape=conditions_shape)

        self.output_projector.units = xz_shape[-1]
        input_shape = list(xz_shape)

        # construct time vector
        input_shape[-1] += 1
        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.subnet.build(input_shape)
        out_shape = self.subnet.compute_output_shape(input_shape)
        self.output_projector.build(out_shape)

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

    def velocity(self, xz: Tensor, t: float | Tensor, conditions: Tensor = None, training: bool = False) -> Tensor:
        if not keras.ops.is_tensor(t):
            t = keras.ops.convert_to_tensor(t, dtype=keras.ops.dtype(xz))

        if keras.ops.ndim(t) == 0:
            t = keras.ops.full((keras.ops.shape(xz)[0],), t, dtype=keras.ops.dtype(xz))

        t = expand_right_as(t, xz)
        t = keras.ops.tile(t, [1] + list(keras.ops.shape(xz)[1:-1]) + [1])

        if conditions is None:
            xtc = keras.ops.concatenate([xz, t], axis=-1)
        else:
            xtc = keras.ops.concatenate([xz, t, conditions], axis=-1)

        return self.output_projector(self.subnet(xtc, training=training), training=training)

    def _velocity_trace(
        self, xz: Tensor, t: Tensor, conditions: Tensor = None, max_steps: int = None, training: bool = False
    ) -> (Tensor, Tensor):
        def f(x):
            return self.velocity(x, t, conditions=conditions, training=training)

        v, trace = jacobian_trace(f, xz, max_steps=max_steps, seed=self.seed_generator, return_output=True)

        return v, keras.ops.expand_dims(trace, axis=-1)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if density:

            def deltas(t, xz):
                v, trace = self._velocity_trace(xz, t, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {"xz": x, "trace": keras.ops.zeros(keras.ops.shape(x)[:-1] + (1,), dtype=keras.ops.dtype(x))}
            state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **kwargs)

            return state["xz"], keras.ops.squeeze(state["trace"], axis=-1)

        def deltas(t, xz):
            return {"xz": self.velocity(xz, t, conditions=conditions, training=training)}

        state = {"xz": x}
        state = integrate(deltas, state, start_time=1.0, stop_time=0.0, **kwargs)

        return state["xz"]

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if density:

            def deltas(t, xz):
                v, trace = self._velocity_trace(xz, t, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {"xz": z, "trace": keras.ops.zeros(keras.ops.shape(z)[:-1] + (1,), dtype=keras.ops.dtype(z))}
            state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **kwargs)

            return state["xz"], keras.ops.squeeze(state["trace"], axis=-1)

        def deltas(t, xz):
            return {"xz": self.velocity(xz, t, conditions=conditions, training=training)}

        state = {"xz": z}
        state = integrate(deltas, state, start_time=0.0, stop_time=1.0, **kwargs)

        return state["xz"]

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

        predicted_velocity = self.velocity(x, t, conditions, training=stage == "training")

        loss = keras.losses.mean_squared_error(target_velocity, predicted_velocity)
        loss = keras.ops.mean(loss)

        return base_metrics | {"loss": loss}
