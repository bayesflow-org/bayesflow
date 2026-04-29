"""
PyTensor → JAX dispatch registrations for the generic log-prob Ops.

Kept in a dedicated module so neither JAXRatio nor JAXContinuous own them —
the Ops are approximator-agnostic and the registrations should fire once,
regardless of which backend class is instantiated.

Imported automatically by ``bayesflow.wrappers.pymc.backend`` at package load.
"""

from pytensor.link.jax.dispatch import jax_funcify

from ..pytensor_ops import RatioLogpOp, RatioLogpVJPOp


@jax_funcify.register(RatioLogpOp)
def logp_op_jax_funcify(op, **kwargs):
    return op.logp_nojit


@jax_funcify.register(RatioLogpVJPOp)
def logp_vjp_op_jax_funcify(op, **kwargs):
    logp_vjp_nojit = op.logp_vjp_nojit

    def logp_vjp(*args):
        *data_and_params, gz = args
        return logp_vjp_nojit(*data_and_params, gz=gz)

    return logp_vjp
