import pytest
import numpy as np
from tests.utils import check_combination_simulator_adapter


def test_approximator_log_prob(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    data = simulator.sample((batch_size,))
    batch = adapter(data)

    approximator.build_from_data(batch)

    if approximator.has_distribution:
        log_prob = approximator.log_prob(data)
        assert isinstance(log_prob, np.ndarray)
        assert log_prob.shape == (batch_size,)
    else:
        with pytest.raises(ValueError):
            approximator.log_prob(data)
