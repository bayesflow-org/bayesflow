from . import (
    approximators,
    adapters,
    datasets,
    diagnostics,
    distributions,
    experimental,
    networks,
    simulators,
    workflows,
    wrappers,
    utils,
)

from .adapters import Adapter
from .approximators import ContinuousApproximator, PointApproximator
from .datasets import OfflineDataset, OnlineDataset, DiskDataset
from .simulators import make_simulator
from .workflows import BasicWorkflow


def setup():
    # perform any necessary setup without polluting the namespace
    import keras
    import logging

    # set the basic logging level if the user hasn't already
    logging.basicConfig(level=logging.INFO)

    # use a separate logger for the bayesflow package
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    from bayesflow.utils import logging

    if keras.backend.backend() == "torch":
        # turn off gradients by default
        import torch

        torch.autograd.set_grad_enabled(False)

        logging.warning("Disabling gradients by default. Use\nwith torch.enable_grad():\nin custom training loops.")

    logging.debug(f"Using backend {keras.backend.backend()!r}")


# call and clean up namespace
setup()
del setup
