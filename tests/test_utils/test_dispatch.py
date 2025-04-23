import keras
import pytest

# Import the dispatch functions
from bayesflow.utils import find_network, find_permutation, find_pooling, find_recurrent_net
from tests.utils import assert_allclose

# --- Tests for find_network.py ---


class DummyMLP:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_find_network_with_string(monkeypatch):
    # Monkeypatch the MLP entry in bayesflow.networks
    monkeypatch.setattr("bayesflow.networks.MLP", DummyMLP)

    net = find_network("mlp", 1, key="value")
    assert isinstance(net, DummyMLP)
    assert net.args == (1,)
    assert net.kwargs == {"key": "value"}


def test_find_network_with_type():
    class CustomNet:
        def __init__(self, x):
            self.x = x

    net = find_network(CustomNet, 42)
    assert isinstance(net, CustomNet)
    assert net.x == 42


def test_find_network_with_keras_layer():
    layer = keras.layers.Dense(10)
    returned = find_network(layer)
    assert returned is layer


def test_find_network_invalid_type():
    with pytest.raises(TypeError):
        find_network(123)


# --- Tests for find_permutation.py ---


class DummyRandomPermutation:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummySwap:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class DummyOrthogonalPermutation:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_find_permutation_random(monkeypatch):
    type("dummy_mod", (), {"RandomPermutation": DummyRandomPermutation})
    monkeypatch.setattr("bayesflow.networks.coupling_flow.permutations.RandomPermutation", DummyRandomPermutation)
    perm = find_permutation("random", 99, flag=True)
    assert isinstance(perm, DummyRandomPermutation)
    assert perm.args == (99,)
    assert perm.kwargs == {"flag": True}


@pytest.mark.parametrize(
    "name,dummy_cls",
    [("swap", DummySwap), ("learnable", DummyOrthogonalPermutation), ("orthogonal", DummyOrthogonalPermutation)],
)
def test_find_permutation_by_name(monkeypatch, name, dummy_cls):
    # Inject dummy classes for each permutation type
    if name == "swap":
        monkeypatch.setattr("bayesflow.networks.coupling_flow.permutations.Swap", dummy_cls)
    else:
        monkeypatch.setattr("bayesflow.networks.coupling_flow.permutations.OrthogonalPermutation", dummy_cls)
    perm = find_permutation(name, "a", b="c")
    assert isinstance(perm, dummy_cls)
    assert perm.args == ("a",)
    assert perm.kwargs == {"b": "c"}


def test_find_permutation_with_keras_layer():
    layer = keras.layers.Activation("relu")
    perm = find_permutation(layer)
    assert perm is layer


def test_find_permutation_with_none():
    res = find_permutation(None)
    assert res is None


def test_find_permutation_invalid_type():
    with pytest.raises(TypeError):
        find_permutation(3.14)


# --- Tests for find_pooling.py ---


def dummy_pooling_constructor(*args, **kwargs):
    return {"args": args, "kwargs": kwargs}


def test_find_pooling_mean():
    pooling = find_pooling("mean")
    # Check that a keras Lambda layer is returned
    assert isinstance(pooling, keras.layers.Lambda)
    # Test that the lambda function produces a mean when applied to a sample tensor.

    sample = keras.ops.convert_to_tensor([[1, 2], [3, 4]])
    # Keras Lambda layers expect tensors via call(), here we simply call the layer's function.
    result = pooling.call(sample)
    assert_allclose(result, keras.ops.mean(sample, axis=-2))


@pytest.mark.parametrize("name,func", [("max", keras.ops.max), ("min", keras.ops.min)])
def test_find_pooling_max_min(name, func):
    pooling = find_pooling(name)
    assert isinstance(pooling, keras.layers.Lambda)

    sample = keras.ops.convert_to_tensor([[1, 2], [3, 4]])
    result = pooling.call(sample)
    assert_allclose(result, func(sample, axis=-2))


def test_find_pooling_learnable(monkeypatch):
    # Monkey patch the PoolingByMultiHeadAttention in its module
    class DummyPoolingAttention:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr("bayesflow.networks.transformers.pma.PoolingByMultiHeadAttention", DummyPoolingAttention)
    pooling = find_pooling("learnable", 7, option="test")
    assert isinstance(pooling, DummyPoolingAttention)
    assert pooling.args == (7,)
    assert pooling.kwargs == {"option": "test"}


def test_find_pooling_with_constructor():
    # Passing a type should result in an instance.
    class DummyPooling:
        def __init__(self, data):
            self.data = data

    pooling = find_pooling(DummyPooling, "dummy")
    assert isinstance(pooling, DummyPooling)
    assert pooling.data == "dummy"


def test_find_pooling_with_keras_layer():
    layer = keras.layers.ReLU()
    pooling = find_pooling(layer)
    assert pooling is layer


def test_find_pooling_invalid_type():
    with pytest.raises(TypeError):
        find_pooling(123)


# --- Tests for find_recurrent_net.py ---


def test_find_recurrent_net_lstm():
    constructor = find_recurrent_net("lstm")
    assert constructor is keras.layers.LSTM


def test_find_recurrent_net_gru():
    constructor = find_recurrent_net("gru")
    assert constructor is keras.layers.GRU


def test_find_recurrent_net_with_keras_layer():
    layer = keras.layers.SimpleRNN(5)
    net = find_recurrent_net(layer)
    assert net is layer


def test_find_recurrent_net_invalid_name():
    with pytest.raises(ValueError):
        find_recurrent_net("invalid_net")


def test_find_recurrent_net_invalid_type():
    with pytest.raises(TypeError):
        find_recurrent_net(3.1415)
