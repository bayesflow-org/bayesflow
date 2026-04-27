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

from importlib import import_module
from bayesflow.utils.logging import warning


def __getattr__(name: str):
    if name == "mamba":
        try:
            module = import_module(f"{__name__}.mamba")
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] == "mamba_ssm":
                warning(
                    "The 'bayesflow.wrappers.mamba' submodule requires the "
                    "'mamba-ssm' package. Install it with:\n\n"
                    "    pip install mamba-ssm\n",
                )
                raise ImportError("Could not import 'bayesflow.wrappers.mamba': mamba-ssm is not installed.") from exc
            raise

        globals()[name] = module
        return module

    if name == "pymc":
        try:
            module = import_module(f"{__name__}.pymc")
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] in {"pymc", "pytensor"}:
                warning(
                    "The 'bayesflow.wrappers.pymc' submodule requires PyMC "
                    "and PyTensor. Install them with:\n\n"
                    "    pip install pymc\n",
                )
                raise ImportError("Could not import 'bayesflow.wrappers.pymc': PyMC is not installed.") from exc
            raise

        globals()[name] = module
        return module

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | {"mamba", "pymc"})


__all__ = ["mamba", "pymc"]
