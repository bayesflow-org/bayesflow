import pytest
from bayesflow.networks import PointNetwork, MLP


def test_default_subnet():
    net = PointNetwork(["mean"], subnet="mlp")
    assert net.subnet is not None


def test_subnet_kwargs_forwarded():
    net = PointNetwork(["quantiles", "mean"], subnet="mlp", subnet_kwargs={"activation": "relu"})
    assert net.subnet.activation == "relu"


def test_explicit_subnet_ignores_kwargs():
    net = PointNetwork(["quantiles"], q=[0.1, 0.5, 0.7, 0.9], subnet=MLP(widths=[2, 3], activation="sigmoid"))
    assert net.subnet.activation == "sigmoid"
    assert net.subnet.widths == [2, 3]


def test_invalid_points_arg():
    with pytest.raises(ValueError) as excinfo:
        PointNetwork(["not-supported"])

    assert "must be either" in str(excinfo)
