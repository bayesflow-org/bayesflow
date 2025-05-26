from . import (
    approximators,
    adapters,
    datasets,
    diagnostics,
    distributions,
    experimental,
    networks,
    simulators,
    utils,
    workflows,
    wrappers,
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

    logging.info(f"Using backend {keras.backend.backend()!r}")

    if keras.backend.backend() == "torch":
        import torch

        torch.autograd.set_grad_enabled(False)

        logging.warning(
            "\n"
            "When using torch backend, we need to disable autograd by default to avoid excessive memory usage. Use\n"
            "\n"
            "with torch.enable_grad():\n"
            "    ...\n"
            "\n"
            "in contexts where you need gradients (e.g. custom training loops)."
        )

    # dynamically add __version__ attribute
    from importlib.metadata import version

    globals()["__version__"] = version("bayesflow")


# call and clean up namespace
setup()
del setup
