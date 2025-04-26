import pytest

import keras


@pytest.fixture(params=[2])
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 3])
def num_features(request):
    return request.param


@pytest.fixture()
def random_samples(batch_size, num_features):
    return keras.random.normal((batch_size, num_features))


@pytest.fixture()
def diagonal_normal():
    from bayesflow.distributions import DiagonalNormal

    return DiagonalNormal()


@pytest.fixture()
def diagonal_student_t():
    from bayesflow.distributions import DiagonalStudentT

    return DiagonalStudentT(df=10)


@pytest.fixture()
def mixture():
    from bayesflow.distributions import DiagonalNormal, DiagonalStudentT, Mixture

    return Mixture([DiagonalNormal(), DiagonalStudentT(df=25)])


@pytest.fixture(params=["diagonal_normal", "diagonal_student_t", "mixture"])
def distribution(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def shape(batch_size, num_features):
    return batch_size, num_features
