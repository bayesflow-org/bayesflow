import keras

from functools import singledispatch


@singledispatch
def find_network(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer network from {arg!r}.")


@find_network.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "mlp" | "default":
            from bayesflow.networks import MLP

            network = MLP(**kwargs)
        # TODO - remove, since MLP encompasses the functionality
        case "resnet":
            from bayesflow.networks import ResNet

            network = ResNet(*args, **kwargs)
        case other:
            raise ValueError(f"Unsupported network name: '{other}'.")

    return network


@find_network.register
def _(network: keras.Layer, *args, **kwargs):
    return network


@find_network.register
def _(constructor: type, *args, **kwargs):
    return constructor(*args, **kwargs)
