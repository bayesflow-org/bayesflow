import pytest

from bayesflow.networks.helpers import DenseBlock, ConditionalDenseBlock


@pytest.fixture()
def dense_block():
    return DenseBlock(width=8)


@pytest.fixture()
def cond_dense_block():
    return ConditionalDenseBlock(width=8)


@pytest.fixture()
def dense_build_shapes():
    return {"input_shape": (4, 6)}


@pytest.fixture()
def cond_build_shapes():
    return {"input_shape": ((4, 6), (4, 3))}
