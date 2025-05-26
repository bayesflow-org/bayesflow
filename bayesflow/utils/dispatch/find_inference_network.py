from functools import singledispatch
import keras


@singledispatch
def find_inference_network(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer inference network from {arg!r}.")


@find_inference_network.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "coupling_flow":
            from bayesflow.networks import CouplingFlow

            return CouplingFlow(*args, **kwargs)

        case "flow_matching":
            from bayesflow.networks import FlowMatching

            return FlowMatching(*args, **kwargs)

        case "consistency_model":
            from bayesflow.networks import ConsistencyModel

            return ConsistencyModel(*args, **kwargs)

        case unknown_network:
            raise ValueError(f"Unknown inference network: '{unknown_network}'")


@find_inference_network.register
def _(cls: type, *args, **kwargs):
    # Instantiate class with the given arguments
    network = cls(*args, **kwargs)
    return network


@find_inference_network.register
def _(layer: keras.Layer, *args, **kwargs):
    return layer


@find_inference_network.register
def _(model: keras.Model, *args, **kwargs):
    return model
