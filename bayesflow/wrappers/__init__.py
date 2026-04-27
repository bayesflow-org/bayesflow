r"""
Optional integrations with third-party libraries.

Submodules are imported lazily — neither :py:mod:`~bayesflow.wrappers.mamba` nor
:py:mod:`~bayesflow.wrappers.pymc` is loaded when you ``import bayesflow``.
A descriptive error is raised if the required dependency is missing.

Submodules
----------
mamba
    :py:class:`~bayesflow.wrappers.mamba.Mamba` summary network backed by the
    ``mamba-ssm`` state-space library.  Install with ``pip install mamba-ssm``.
pymc
    :py:class:`~bayesflow.wrappers.pymc.RatioDistribution` — use a trained
    :py:class:`~bayesflow.approximators.RatioApproximator` as a likelihood term
    inside a PyMC model.  Install with ``pip install pymc``.

Examples
--------
>>> from bayesflow.wrappers.pymc import RatioDistribution  # doctest: +SKIP
>>> ratio_dist = RatioDistribution(approximator, param_names=["mu", "sigma"])  # doctest: +SKIP
"""


def __getattr__(name: str):
    if name == "mamba":
        try:
            from . import mamba as _mamba

            globals()["mamba"] = _mamba
            return _mamba
        except ImportError as exc:
            import warnings

            warnings.warn(
                "The 'bayesflow.wrappers.mamba' submodule requires the 'mamba-ssm' package. "
                "Install it with:\n\n    pip install mamba-ssm\n",
                ImportWarning,
                stacklevel=2,
            )
            raise ImportError("Could not import 'bayesflow.wrappers.mamba': mamba-ssm is not installed.") from exc

    if name == "pymc":
        try:
            from . import pymc as _pymc

            globals()["pymc"] = _pymc
            return _pymc
        except ImportError as exc:
            import warnings

            warnings.warn(
                "The 'bayesflow.wrappers.pymc' submodule requires PyMC and PyTensor. "
                "Install them with:\n\n    pip install pymc\n",
                ImportWarning,
                stacklevel=2,
            )
            raise ImportError("Could not import 'bayesflow.wrappers.pymc': PyMC is not installed.") from exc

    raise AttributeError(f"module 'bayesflow.wrappers' has no attribute {name!r}")
