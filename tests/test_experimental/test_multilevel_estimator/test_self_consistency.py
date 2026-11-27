import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.experimental.multilevel_estimator import MultilevelDataset, MultilevelEstimator
from bayesflow.experimental.self_consistency import MultiDataset, SelfConsistencyLoss, SemiSupervisedEstimator
from bayesflow.networks import CouplingFlow

from .conftest import estimator_for


@pytest.fixture(params=["two_level_simulator", "survey_simulator"])
def simulator(request):
    return request.getfixturevalue(request.param)


class ZeroDensity:
    def __init__(self, graph):
        self.root = next(node for node in graph if graph.in_degree(node) == 0)

    def log_prob(self, **batch):
        return keras.ops.zeros(keras.ops.shape(batch[self.root]["indices"][self.root])[0])


def densities(simulator):
    if "hypers" in simulator.graph:
        return {"prior": simulator.prior, "likelihood": simulator.likelihood}

    density = ZeroDensity(simulator.graph)

    return {"prior": density, "likelihood": density}


def test_build_from_shapes_matches_build_from_data(simulator):
    batch = estimator_for(simulator).to_batch(simulator.sample(2))

    from_data = estimator_for(simulator)
    from_data.build_from_data(batch)
    from_shapes = estimator_for(simulator)
    from_shapes.build(keras.tree.map_structure(keras.ops.shape, batch))

    assert from_shapes.built
    assert from_shapes.variable_dims == from_data.variable_dims


def test_fit_with_unbuilt_posterior(simulator):
    graph = simulator.graph.expand().invert()
    adapter = Adapter().log(["hyper_std", "shared_std"]) if "hypers" in simulator.graph else Adapter()
    posterior = MultilevelEstimator(graph, inference_networks=[CouplingFlow() for _ in graph.networks()], adapter=adapter)
    estimator = SemiSupervisedEstimator(estimators=densities(simulator) | {"posterior": posterior})
    estimator.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        estimator_metrics={"labeled": ["posterior"]},
        composite_metrics={"unlabeled": SelfConsistencyLoss(num_samples=3)},
    )
    dataset = MultiDataset(
        labeled=MultilevelDataset(simulator.sample(8), posterior.graph, batch_size=4),
        unlabeled=MultilevelDataset(simulator.sample(2), posterior.graph, batch_size=2, padded=False),
    )

    history = estimator.fit(dataset=dataset, epochs=1, verbose=0)

    assert posterior.built
    assert all(np.all(np.isfinite(values)) for values in history.history.values())
