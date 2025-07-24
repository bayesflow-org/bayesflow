# ruff: noqa: E402
# disable E402 to allow for setup code before importing any internals (which could import keras)


def setup():
    # perform any necessary setup without polluting the namespace
    import os
    from importlib.util import find_spec

    issue_url = "https://github.com/bayesflow-org/bayesflow/issues/new?template=bug_report.md"

    if "KERAS_BACKEND" not in os.environ:
        # check for available backends and automatically set the KERAS_BACKEND env variable or raise an error
        class Backend:
            def __init__(self, display_name, package_name, env_name, install_url, priority):
                self.display_name = display_name
                self.package_name = package_name
                self.env_name = env_name
                self.install_url = install_url
                self.priority = priority

        backends = [
            Backend("JAX", "jax", "jax", "https://docs.jax.dev/en/latest/quickstart.html#installation", 0),
            Backend("PyTorch", "torch", "torch", "https://pytorch.org/get-started/locally/", 1),
            Backend("TensorFlow", "tensorflow", "tensorflow", "https://www.tensorflow.org/install", 2),
        ]

        found_backends = []
        for backend in backends:
            if find_spec(backend.package_name) is not None:
                found_backends.append(backend)

        if not found_backends:
            message = "No suitable backend found. Please install one of the following:\n"
            for backend in backends:
                message += f"{backend.display_name}\n"
            message += "\n"

            message += f"If you continue to see this error, please file a bug report at {issue_url}.\n"
            message += (
                "You can manually select a backend by setting the KERAS_BACKEND environment variable as shown below:\n"
            )
            message += "https://keras.io/getting_started/#configuring-your-backend"

            raise ImportError(message)

        if len(found_backends) > 1:
            import warnings

            found_backends.sort(key=lambda b: b.priority)
            chosen_backend = found_backends[0]

            warnings.warn(
                f"Multiple Keras-compatible backends detected ({', '.join(b.display_name for b in found_backends)}).\n"
                f"Defaulting to {chosen_backend.display_name}.\n"
                "To override, set the KERAS_BACKEND environment variable before importing bayesflow.\n"
                "See: https://keras.io/getting_started/#configuring-your-backend"
            )
        else:
            os.environ["KERAS_BACKEND"] = found_backends[0].env_name

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

from . import (
    approximators,
    adapters,
    augmentations,
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
