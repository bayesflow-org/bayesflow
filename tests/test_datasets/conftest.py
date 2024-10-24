import keras
import numpy as np
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture()
def num_batches():
    return 4


@pytest.fixture(params=["online_dataset", "offline_dataset"])
def dataset(request, online_dataset, offline_dataset):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def model():
    class Model(keras.Model):
        def call(self, *args, **kwargs):
            pass

        def compute_loss(self, *args, **kwargs):
            return keras.ops.zeros(())

    model = Model()
    model.compile()

    return model


@pytest.fixture()
def offline_dataset(simulator, batch_size, num_batches, workers, use_multiprocessing):
    from bayesflow import OfflineDataset

    # TODO: there is a bug in keras where if len(dataset) == 1 batch
    #  fit will error because no logs are generated
    #  the single batch is then skipped entirely
    data = simulator.sample((batch_size * num_batches,))
    return OfflineDataset(
        data, batch_size=batch_size, workers=workers, use_multiprocessing=use_multiprocessing, adapter=None
    )


@pytest.fixture()
def online_dataset(simulator, batch_size, num_batches, workers, use_multiprocessing):
    from bayesflow import OnlineDataset

    return OnlineDataset(
        simulator,
        batch_size=batch_size,
        num_batches=num_batches,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        adapter=None,
    )


# these need to be global for pickle


class Simulator:
    def sample(self, batch_shape):
        return dict(x=np.random.standard_normal(size=batch_shape + (2,)).astype("float32"))


def sample_contexts_unbatched(**kwargs):
    return dict(r=np.random.standard_normal(), alpha=np.random.standard_normal())


def sample_parameters_unbatched(**kwargs):
    return dict(theta=np.random.standard_normal(size=2))


def sample_observables_unbatched(r, alpha, theta, **kwargs):
    return dict(x=np.random.standard_normal(size=2))


def sample_contexts_batched(shape, **kwargs):
    return dict(r=np.random.standard_normal(size=shape), alpha=np.random.standard_normal(size=shape))


def sample_parameters_batched(shape, **kwargs):
    return dict(theta=np.random.standard_normal(size=shape + (2,)))


def sample_observables_batched(shape, r, alpha, theta, **kwargs):
    return dict(x=np.random.standard_normal(size=shape + (2,)))


@pytest.fixture(params=["class", "unbatched_composite"])
def simulator(request):
    from bayesflow.simulators import make_simulator

    if request.param == "class":
        simulator = Simulator()
    elif request.param == "unbatched_composite":
        simulator = make_simulator(
            [sample_contexts_unbatched, sample_parameters_unbatched, sample_observables_unbatched]
        )
    else:
        raise NotImplementedError

    return simulator


@pytest.fixture(params=[False])
def use_multiprocessing(request):
    return request.param


@pytest.fixture(params=[1, 2])
def workers(request):
    return request.param
