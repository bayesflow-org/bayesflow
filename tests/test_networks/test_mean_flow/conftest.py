import keras
import numpy as np
import pytest


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[2], scope="session")
def xz_dim(request):
    return request.param


@pytest.fixture(params=[None, 3], scope="session")
def cond_dim(request):
    return request.param


@pytest.fixture(scope="session")
def random_samples(batch_size, xz_dim):
    return keras.random.normal((batch_size, xz_dim))


@pytest.fixture(scope="session")
def random_conditions(batch_size, cond_dim):
    if cond_dim is None:
        return None
    return keras.random.normal((batch_size, cond_dim))


@pytest.fixture(
    params=[
        dict(),
        dict(velocity_head_layers=0),
        dict(cycle_weight=0.0),
        dict(use_optimal_transport=True),
        dict(mask_handler=dict(fixed_target_prob=0.3, missing_target_prob=0.3, missing_conditions_prob=0.3)),
        dict(density_head=True, mask_handler=dict(fixed_target_prob=0.3)),
    ],
    ids=["default", "no_head", "no_cycle", "ot", "masking", "density_head"],
)
def mean_flow(request):
    from bayesflow.experimental import MeanFlow
    from bayesflow.masking import MaskHandler

    kwargs = dict(request.param)
    if "mask_handler" in kwargs:
        kwargs["mask_handler"] = MaskHandler(**kwargs["mask_handler"])
    return MeanFlow(subnet_kwargs=dict(widths=(8, 8)), **kwargs)


def build(network, random_samples, random_conditions):
    xz_shape = keras.ops.shape(random_samples)
    cond_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
    network.build(xz_shape, conditions_shape=cond_shape)
    return network


def make_mean_flow(random_samples, random_conditions, **kwargs):
    from bayesflow.experimental import MeanFlow

    kwargs.setdefault("subnet_kwargs", dict(widths=(8, 8)))
    return build(MeanFlow(**kwargs), random_samples, random_conditions)


def make_transformer_mean_flow(random_samples, random_conditions, **kwargs):
    network = make_mean_flow(
        random_samples,
        random_conditions,
        subnet="diffusion_transformer",
        subnet_kwargs=dict(widths=(8, 8), num_heads=2),
        **kwargs,
    )

    # the readouts are zero-initialized, so an untrained network would predict zero
    readouts = (network.subnet, network.velocity_readout, network.divergence_readout)
    for readout in filter(None, readouts):
        for weight in readout.token_out.weights + readout.final_ada_ln.weights:
            weight.assign(keras.random.normal(weight.shape, stddev=0.3))
    return network


def max_abs_diff(x1, x2):
    return np.max(np.abs(keras.ops.convert_to_numpy(x1) - keras.ops.convert_to_numpy(x2)))
