from functools import partial

import keras
import numpy as np
import pytest


NETWORK_CASES = [
    "diffusion",
    "flow_matching",
    "consistency",
    pytest.param("stable_consistency", marks=pytest.mark.skip_on_mps),
]


@pytest.fixture(autouse=True)
def random_seed():
    """Override the repository fixture without mutating Keras' global RNG."""
    return 1729


def make_network(name, *, subnet="time_mlp", **kwargs):
    from bayesflow.networks import ConsistencyModel, DiffusionModel, FlowMatching, StableConsistencyModel

    subnet_kwargs = {"widths": (8, 8)} if isinstance(subnet, str) else None
    common = {"subnet": subnet, "subnet_kwargs": subnet_kwargs} | kwargs

    if name == "diffusion":
        integrate_kwargs = common.pop("integrate_kwargs", {"method": "euler", "steps": 2})
        return DiffusionModel(integrate_kwargs=integrate_kwargs, **common)
    if name == "flow_matching":
        integrate_kwargs = common.pop("integrate_kwargs", {"method": "euler", "steps": 2})
        return FlowMatching(integrate_kwargs=integrate_kwargs, **common)
    if name == "consistency":
        return ConsistencyModel(total_steps=4, s0=2, s1=4, max_time=2.0, **common)
    if name == "stable_consistency":
        return StableConsistencyModel(steps=2, weight_mlp_kwargs={"widths": (8, 8)}, **common)
    raise ValueError(f"Unknown masking test network: {name!r}")


@pytest.fixture(params=NETWORK_CASES)
def network_name(request):
    return request.param


@pytest.fixture
def network_factory(network_name):
    return partial(make_network, network_name)


@pytest.fixture
def targets():
    return keras.ops.convert_to_tensor(
        np.asarray(
            [
                [-1.0, 0.5, 2.0],
                [0.25, -0.75, 1.25],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def conditions():
    return keras.ops.convert_to_tensor(
        np.asarray(
            [
                [0.5, -1.5],
                [1.0, 0.25],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def fixed_target_mask():
    return keras.ops.convert_to_tensor(
        np.asarray(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def infer_target_mask():
    return keras.ops.convert_to_tensor(
        np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def observed_condition_mask():
    return keras.ops.convert_to_tensor(
        np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
