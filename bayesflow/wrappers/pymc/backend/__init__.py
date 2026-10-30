import keras

from . import jax_dispatch  # noqa: F401 — registers pytensor→JAX dispatch for the log-prob Ops


def build_backend(estimator, param_names, *, exchangeable: bool = True):
    """Return the appropriate JAX backend for *estimator*."""
    if keras.backend.backend() != "jax":
        raise ImportError(f"Backend '{keras.backend.backend()}' is not yet supported. Please use 'jax'.")

    from bayesflow.estimators import RatioEstimator, BasicEstimator

    if isinstance(estimator, RatioEstimator):
        from .jax_ratio import JAXRatio

        return JAXRatio(estimator, param_names, exchangeable=exchangeable)

    if isinstance(estimator, BasicEstimator):
        from .jax_continuous import JAXContinuous

        return JAXContinuous(estimator, param_names, exchangeable=exchangeable)

    raise TypeError(
        f"No JAX backend available for estimator type '{type(estimator).__name__}'. "
        "Expected RatioEstimator (NRE) or BasicEstimator (NLE)."
    )


__all__ = ["build_backend"]
