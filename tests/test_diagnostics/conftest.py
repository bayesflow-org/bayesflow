import keras
import numpy as np
import pytest
from bayesflow.utils.numpy_utils import softmax


@pytest.fixture()
def var_names():
    return [r"$\beta_0$", r"$\beta_1$", r"$\sigma$"]


@pytest.fixture()
def random_samples_a():
    return np.random.normal(loc=0, scale=1, size=(5000, 8))


@pytest.fixture()
def random_samples_b():
    return np.random.normal(loc=0, scale=3, size=(5000, 8))


@pytest.fixture()
def random_estimates():
    return {
        "beta": np.random.standard_normal(size=(32, 10, 2)),
        "sigma": np.random.standard_normal(size=(32, 10, 1)),
    }


@pytest.fixture()
def random_targets():
    return {
        "beta": np.random.standard_normal(size=(32, 2)),
        "sigma": np.random.standard_normal(size=(32, 1)),
        "y": np.random.standard_normal(size=(32, 3, 1)),
    }


@pytest.fixture()
def random_priors():
    return {
        "beta": np.random.standard_normal(size=(64, 2)),
        "sigma": np.random.standard_normal(size=(64, 1)),
        "y": np.random.standard_normal(size=(64, 3, 1)),
    }


@pytest.fixture()
def model_names():
    return [r"$\mathcal{M}_0$", r"$\mathcal{M}_1$", r"$\mathcal{M}_2$"]


@pytest.fixture()
def true_models():
    true_models = np.random.choice(3, 100)
    true_models = np.eye(3)[true_models].astype(np.int32)
    return true_models


@pytest.fixture()
def pred_models(true_models):
    pred_models = np.random.normal(loc=true_models)
    pred_models = softmax(pred_models, axis=-1)
    return pred_models


@pytest.fixture()
def history():
    h = keras.callbacks.History()

    step = np.linspace(0, 1, 10_000)
    train_loss = (1.0 - step) ** 2 + np.random.normal(loc=0, scale=0.02, size=step.shape)
    validation_loss = 0.1 + (0.75 - step) ** 2 + np.random.normal(loc=0, scale=0.02, size=step.shape)

    h.history = {
        "loss": train_loss.tolist(),
        "val_loss": validation_loss.tolist(),
    }

    return h


@pytest.fixture()
def adapter():
    from bayesflow.adapters import Adapter

    return Adapter.create_default("parameters").rename("observables", "summary_variables")


@pytest.fixture()
def summary_network():
    from tests.utils import MeanStdSummaryNetwork

    return MeanStdSummaryNetwork()
