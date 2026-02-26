import pytest
from bayesflow.networks import PointNetwork, MLP


def assert_subnet_has(**expected):
    def _assert(subnet_layer):
        for k, v in expected.items():
            assert getattr(subnet_layer, k) == v

    return _assert


@pytest.mark.parametrize(
    "points,q,subnet,kwargs,assertion",
    [
        (["mean"], None, "mlp", {}, lambda arg: True),
        (
            ["quantiles", "mean"],
            None,
            "mlp",
            {"subnet_kwargs": {"activation": "relu"}},
            assert_subnet_has(activation="relu"),
        ),
        (
            ["quantiles"],
            [0.1, 0.5, 0.7, 0.9],
            MLP(widths=[2, 3], activation="sigmoid"),
            {"subnet_kwargs": {"activation": "relu"}},  # should be ignored
            assert_subnet_has(activation="sigmoid", widths=[2, 3]),
        ),
    ],
)
def test_kwargs_forwarded(points, q, subnet, kwargs, assertion):
    net = PointNetwork(points, q, subnet, **kwargs)
    subnet_layer = net.subnet  # adjust path
    assertion(subnet_layer)
