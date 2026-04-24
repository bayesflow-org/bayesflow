import keras

from pytest import fixture


@fixture(params=[keras.ops.square])
def fn_unary_scalar(request):
    return request.param


@fixture(params=[keras.ops.sum])
def fn_unary_vector(request):
    return request.param


@fixture(params=[keras.ops.multiply])
def fn_binary_scalars(request):
    return request.param


@fixture(params=[keras.ops.dot])
def fn_binary_vectors(request):
    return request.param


@fixture(params=[False, True])
def jit_compile(request):
    return request.param
