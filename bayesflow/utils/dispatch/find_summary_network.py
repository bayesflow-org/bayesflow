from functools import singledispatch
import keras


@singledispatch
def find_summary_network(arg, *args, **kwargs):
    raise TypeError(f"Cannot infer inference network from {arg!r}.")


@find_summary_network.register
def _(name: str, *args, **kwargs):
    match name.lower():
        case "deep_set":
            from bayesflow.networks import DeepSet

            return DeepSet(*args, **kwargs)

        case "set_transformer":
            from bayesflow.networks import SetTransformer

            return SetTransformer(*args, **kwargs)

        case "fusion_transformer":
            from bayesflow.networks import FusionTransformer

            return FusionTransformer(*args, **kwargs)

        case "time_series_transformer":
            from bayesflow.networks import TimeSeriesTransformer

            return TimeSeriesTransformer(*args, **kwargs)

        case "time_series_network":
            from bayesflow.networks import TimeSeriesNetwork

            return TimeSeriesNetwork(*args, **kwargs)

        case unknown_network:
            raise ValueError(f"Unknown summary network: '{unknown_network}'")


@find_summary_network.register
def _(cls: type, *args, **kwargs):
    # Instantiate class with the given arguments
    network = cls(*args, **kwargs)
    return network


@find_summary_network.register
def _(layer: keras.Layer, *args, **kwargs):
    return layer


@find_summary_network.register
def _(model: keras.Model, *args, **kwargs):
    return model
