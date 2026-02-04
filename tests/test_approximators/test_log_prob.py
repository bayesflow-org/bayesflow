import pytest
import numpy as np
from tests.utils import check_combination_simulator_adapter

from bayesflow import OnlineDataset, EnsembleDataset, EnsembleApproximator


def test_approximator_log_prob(approximator, simulator, batch_size, adapter):
    check_combination_simulator_adapter(simulator, adapter)

    data = simulator.sample((batch_size,))

    train_dataset = OnlineDataset(simulator=simulator, adapter=adapter, num_batches=4, batch_size=batch_size)
    if isinstance(approximator, EnsembleApproximator):
        train_dataset = EnsembleDataset(train_dataset, ensemble_size=len(approximator.approximators))
    batch = train_dataset[0]

    approximator.build_from_data(batch)

    if approximator.has_distribution:
        log_prob = approximator.log_prob(data)
        assert isinstance(log_prob, np.ndarray)
        assert log_prob.shape == (batch_size,)
    else:
        with pytest.raises(ValueError):
            approximator.log_prob(data)
