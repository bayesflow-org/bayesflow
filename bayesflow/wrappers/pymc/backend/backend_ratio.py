import keras


match keras.config.backend():
    case "jax":
        from .jax_ratio import JAXRatio as BackendRatio
    case other:
        raise ImportError(f"Backend '{other}' is not yet supported. Please, use 'jax'.")


__all__ = ["BackendRatio"]
