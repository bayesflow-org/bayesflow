import keras


match keras.backend.backend():
    case "jax":
        pass
    case other:
        raise ValueError(f"Backend '{other}' is not supported.")
