"""Frozen default configuration dicts for inference network subnets and solvers.

All values are wrapped in :class:`~types.MappingProxyType` to prevent
accidental mutation.  Use the ``|`` operator to merge with user overrides::

    from bayesflow.networks.defaults import TIME_MLP_DEFAULTS

    subnet_kwargs = TIME_MLP_DEFAULTS | (user_kwargs or {})
"""

from types import MappingProxyType

TIME_MLP_DEFAULTS = MappingProxyType(
    {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
        "time_embedding_dim": 32,
        "merge": "concat",
        "norm": "layer",
    }
)

WEIGHT_MLP_DEFAULTS = MappingProxyType(
    {
        "widths": (256,),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": False,
        "dropout": 0.05,
        "spectral_normalization": False,
    }
)

COUPLING_MLP_DEFAULTS = MappingProxyType(
    {
        "activation": "hard_silu",
        "kernel_initializer": "glorot_uniform",
        "residual": False,
        "dropout": 0.05,
        "spectral_normalization": False,
    }
)

DIFFUSION_INTEGRATE_DEFAULTS = MappingProxyType({"method": "two_step_adaptive", "steps": "adaptive"})

FLOW_MATCHING_INTEGRATE_DEFAULTS = MappingProxyType({"method": "tsit5", "steps": "adaptive"})

OPTIMAL_TRANSPORT_DEFAULTS = MappingProxyType(
    {
        "method": "log_sinkhorn",
        "regularization": 0.1,
        "max_steps": 100,
        "atol": 1e-5,
        "partial_factor": 1.0,
        "condition_ratio": 0.01,
    }
)
