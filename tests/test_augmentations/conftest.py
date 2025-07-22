import numpy as np
import pytest


@pytest.fixture()
def random_data():
    return {
        "x1": np.random.standard_normal(size=(4, 1)),
        "x2": np.random.standard_normal(size=(8, 10, 1)),
    }
