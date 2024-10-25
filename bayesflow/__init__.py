from . import (
    approximators,
    benchmarks,
    data_adapters,
    datasets,
    diagnostics,
    distributions,
    networks,
    simulators,
    utils,
)

from .approximators import ContinuousApproximator
from .data_adapters import DataAdapter
from .datasets import OfflineDataset, OnlineDataset, DiskDataset
from .simulators import make_simulator


def setup():
    # perform any necessary setup without polluting the namespace
    import keras
    import logging

    # set the basic logging level if the user hasn't already
    logging.basicConfig(level=logging.INFO)

    # use a separate logger for the bayesflow package
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if keras.backend.backend() == "torch":
        # turn off gradients by default
        import torch

        torch.autograd.set_grad_enabled(False)


# call and clean up namespace
setup()
del setup
