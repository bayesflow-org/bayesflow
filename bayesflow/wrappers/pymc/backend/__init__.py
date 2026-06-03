import keras

from . import jax_dispatch  # noqa: F401 — registers pytensor→JAX dispatch for the log-prob Ops


def build_backend(approximator, param_names, *, exchangeable: bool = True):
    """Return the appropriate JAX backend for *approximator*."""
    if keras.backend.backend() != "jax":
        raise ImportError(f"Backend '{keras.backend.backend()}' is not yet supported. Please use 'jax'.")

    from bayesflow.approximators import RatioApproximator, ContinuousApproximator

    if isinstance(approximator, RatioApproximator):
        from .jax_ratio import JAXRatio

        return JAXRatio(approximator, param_names, exchangeable=exchangeable)

    if isinstance(approximator, ContinuousApproximator):
        from .jax_continuous import JAXContinuous

        return JAXContinuous(approximator, param_names, exchangeable=exchangeable)

    raise TypeError(
        f"No JAX backend available for approximator type '{type(approximator).__name__}'. "
        "Expected RatioApproximator (NRE) or ContinuousApproximator (NLE)."
    )


__all__ = ["build_backend"]
