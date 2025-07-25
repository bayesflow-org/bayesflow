from bayesflow.networks import DiffusionModel as StabilizedDiffusionModel


def DiffusionModel(*args, **kwargs):
    from warnings import warn

    warn(
        "DiffusionModel has been stabilized and moved to bayesflow.networks. "
        "Please switch your imports to the new location. This reference will be "
        "removed in a future version.",
        FutureWarning,
    )
    return StabilizedDiffusionModel(*args, **kwargs)
