from collections.abc import Callable

import numpy as np

import pytensor
import pytensor.tensor as pt

from pytensor.graph.basic import Apply
from pytensor.graph.op import Op


class RatioLogpVJPOp(Op):
    """
    PyTensor Op for vector-Jacobian products of the vmapped log-ratio.

    Returns one gradient tensor per distribution parameter (gradients with
    respect to the observed data are not computed).

    Parameters
    ----------
    logp_vjp_jit : Callable
        A JIT-compiled function ``(data, *params, gz) -> tuple[array, ...]``
        that evaluates the VJP of the log-ratio with respect to ``params``.
    """

    def __init__(self, logp_vjp_jit: Callable):
        self.logp_vjp_jit = logp_vjp_jit

    def make_node(self, data, *dist_params, gz):
        data = pt.as_tensor_variable(data)
        dist_params = [pt.as_tensor_variable(p) for p in dist_params]
        gz = pt.as_tensor_variable(gz)

        inputs = [data] + dist_params + [gz]
        outputs = [p.type() for p in dist_params]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        grads = self.logp_vjp_jit(*inputs[:-1], gz=inputs[-1])
        for i, grad in enumerate(grads):
            outputs[i][0] = np.asarray(grad, dtype=node.outputs[i].dtype)


class RatioLogpOp(Op):
    """
    PyTensor Op for the (optionally vmapped) log-ratio over observations.

    Parameters
    ----------
    logp_jit : Callable
        JIT-compiled ``(data, *params) -> array`` that evaluates the
        (vmapped) log-ratio.
    logp_nojit : Callable
        Non-JIT version of the same function, used by JAX-backend
        samplers via ``jax_funcify``.
    vjp_op : RatioLogpVJPOp
        The companion Op that computes parameter gradients.
    scalar_output : bool, optional
        If ``True`` the Op returns a scalar (joint / non-exchangeable
        case).  If ``False`` (default) it returns a vector of per-trial
        log-ratios (exchangeable / i.i.d. case).
    """

    def __init__(
        self,
        logp_jit: Callable,
        logp_nojit: Callable,
        vjp_op: RatioLogpVJPOp,
        *,
        scalar_output: bool = False,
    ):
        self.logp_jit = logp_jit
        self.logp_nojit = logp_nojit
        self.vjp_op = vjp_op
        self.scalar_output = scalar_output

    def make_node(self, data, *dist_params):
        data = pt.as_tensor_variable(data)
        dist_params = [pt.as_tensor_variable(p) for p in dist_params]
        inputs = [data] + dist_params
        if self.scalar_output:
            outputs = [pt.scalar(dtype=pytensor.config.floatX)]
        else:
            outputs = [pt.vector(dtype=pytensor.config.floatX)]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, output_storage):
        result = self.logp_jit(*inputs)
        output_storage[0][0] = np.asarray(result, dtype=node.outputs[0].dtype)

    def grad(self, inputs, output_gradients):
        data, *dist_params = inputs
        gz = output_gradients[0]
        param_grads = self.vjp_op(data, *dist_params, gz=gz)
        return [
            pytensor.gradient.grad_not_implemented(self, 0, data),
            *param_grads,
        ]
