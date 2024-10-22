import keras
from bayesflow.types import Tensor, Shape
from bayesflow.utils import find_network, jacobian_trace, keras_kwargs, expand_right_as, tile_axis
from .integrator import Integrator


class RK2Integrator(Integrator):
    """
    TODO: docstring
    """

    def __init__(self, subnet: str | type = "mlp", **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.subnet = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros", kernel_initializer="zeros")

    def build(self, xz_shape: Shape, conditions_shape: Shape = None):
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

    def velocity(self, x: Tensor, t: int | float | Tensor, conditions: Tensor = None, **kwargs):
        if not keras.ops.is_tensor(t):
            t = keras.ops.convert_to_tensor(t, dtype=x.dtype)
        if keras.ops.ndim(t) == 0:
            t = keras.ops.full((keras.ops.shape(x)[0],), t, dtype=keras.ops.dtype(x))

        t = expand_right_as(t, x)
        if keras.ops.ndim(x) == 3:
            t = tile_axis(t, n=keras.ops.shape(x)[1], axis=1)

        if conditions is None:
            xtc = keras.ops.concatenate([x, t], axis=-1)
        else:
            xtc = keras.ops.concatenate([x, t, conditions], axis=-1)

        return self.output_projector(self.subnet(xtc, **kwargs))

    def call(
        self,
        x: Tensor,
        conditions: Tensor = None,
        steps: int = 100,
        density: bool = False,
        inverse: bool = False,
        **kwargs,
    ):
        z = keras.ops.copy(x)
        t = 1.0 if not inverse else 0.0
        dt = -1.0 / steps if not inverse else 1.0 / steps

        def f(arg):
            k1 = self.velocity(arg, t, conditions, **kwargs)
            k2 = self.velocity(arg + (dt / 2.0 * k1), t + (dt / 2.0), conditions, **kwargs)
            return k2

        if density:
            trace = keras.ops.zeros(keras.ops.shape(x)[:-1], dtype=x.dtype)
            for _ in range(steps):
                k2, tr = jacobian_trace(f, z, kwargs.get("trace_steps", 5))
                z += dt * k2
                trace += dt * tr
                t += dt
            return z, trace

        for _ in range(steps):
            k1 = self.velocity(z, t, conditions, **kwargs)
            k2 = self.velocity(z + (dt / 2.0 * k1), t + (dt / 2.0), conditions, **kwargs)
            z += dt * k2
            t += dt

        return z
