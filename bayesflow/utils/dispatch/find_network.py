from functools import singledispatch
import keras


@singledispatch
def find_network(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer network from {arg!r}.")


@find_network.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "mlp" | "default":
            from bayesflow.networks import MLP

            network = MLP(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported network name: '{other}'.")

    return network


@find_network.register
def _(cls: type, *args, **kwargs):
    # Instantiate class with the given arguments
    network = cls(*args, **kwargs)
    return network


@find_network.register
def _(layer: keras.Layer, *args, **kwargs):
    return layer
