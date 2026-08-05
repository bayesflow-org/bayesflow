import functools

import keras
import numpy as np
import pytest

from bayesflow.utils import filter_kwargs


def _network_class(name):
    from bayesflow.networks import ConsistencyModel, DiffusionModel, FlowMatching, StableConsistencyModel

    return {
        "flow_matching": FlowMatching,
        "consistency_model": ConsistencyModel,
        "stable_consistency_model": StableConsistencyModel,
        "diffusion_model": DiffusionModel,
    }[name]


@pytest.fixture(
    params=[
        "flow_matching",
        "consistency_model",
        "stable_consistency_model",
        "diffusion_model",
        "flow_matching_transformer",
        "consistency_model_transformer",
        "stable_consistency_model_transformer",
        "diffusion_model_transformer",
    ],
    scope="function",
)
def diffusion_type_inference_network(request):
    name, transformer_suffix, _ = request.param.partition("_transformer")
    network = _network_class(name)
    if transformer_suffix:
        network = functools.partial(network, subnet="diffusion_transformer")
    return network


def _skip_torch_cpu_masking_workflow():
    if keras.backend.backend() != "torch":
        return

    import torch

    if torch.cuda.device_count() == 0:
        pytest.skip(
            "PyTorch CPU scaled-dot-product attention does not support forward-mode AD used by the masking workflow."
        )


def _build_network(diffusion_type_inference_network, epochs, num_batches_per_epoch, **extra_kwargs):
    # sampling accuracy is irrelevant here, so ODE-based networks use a few fixed euler steps
    return diffusion_type_inference_network(
        subnet_kwargs=dict(widths=(8, 8)),
        fixed_target_prob=0.5,
        **extra_kwargs,
        **filter_kwargs(
            dict(
                total_steps=epochs * num_batches_per_epoch,
                s0=3,
                s1=10,
                eps=1e-8,
                integrate_kwargs=dict(method="euler", steps=20),
            ),
            diffusion_type_inference_network,
        ),
    )


def test_masking(diffusion_type_inference_network):
    _skip_torch_cpu_masking_workflow()

    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    num_samples = 3
    batch_size = 2
    num_batches_per_epoch = 2
    epochs = 5
    n_test_data = 5
    workflow = BasicWorkflow(
        inference_network=_build_network(
            diffusion_type_inference_network, epochs, num_batches_per_epoch, missing_target_prob=0.5
        ),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=epochs, batch_size=batch_size, num_batches_per_epoch=num_batches_per_epoch)
    test_conditions = workflow.simulate((n_test_data,))
    samples = workflow.sample(num_samples=num_samples, conditions=test_conditions)["parameters"]

    test_conditions_adapted = workflow.adapter(test_conditions)
    fixed_target_mask = keras.ops.concatenate(
        (
            keras.ops.ones(1),  # param 1 is inferred
            keras.ops.zeros(1),  # param 2 is fixed
        )
    )
    fixed_target_mask = np.broadcast_to(fixed_target_mask, (n_test_data, 2))
    targets_fixed = test_conditions_adapted["inference_variables"]

    fixed_samples = workflow.sample(
        conditions=test_conditions,
        num_samples=num_samples,
        fixed_target_value=targets_fixed,
        fixed_target_mask=fixed_target_mask,
    )["parameters"]
    assert samples.shape == fixed_samples.shape
    assert (np.abs(fixed_samples[..., 1] - test_conditions["parameters"][:, 1:]) < 1e-6).all()
    assert (np.abs(fixed_samples[..., 0] - test_conditions["parameters"][:, :1]) > 0.1).any()  # should vary

    infer_target_mask = keras.ops.concatenate(
        (
            keras.ops.ones(1),  # param 1 is inferred only
            keras.ops.zeros(1),  # param 2 is marginalized
        )
    )
    infer_target_mask = np.broadcast_to(infer_target_mask, (5, 2))
    marginalized_samples = workflow.sample(
        conditions=test_conditions,
        num_samples=num_samples,
        infer_target_mask=infer_target_mask,
    )["parameters"]
    assert samples.shape == marginalized_samples.shape


def test_masking_unconditional(diffusion_type_inference_network):
    _skip_torch_cpu_masking_workflow()

    from bayesflow import BasicWorkflow
    from bayesflow.simulators import TwoMoons

    num_samples = 3
    batch_size = 2
    num_batches_per_epoch = 2
    epochs = 5
    workflow = BasicWorkflow(
        inference_network=_build_network(diffusion_type_inference_network, epochs, num_batches_per_epoch),
        inference_variables=["parameters"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=epochs, batch_size=batch_size, num_batches_per_epoch=num_batches_per_epoch)
    test_conditions = workflow.simulate((5,))

    test_conditions_adapted = workflow.adapter(test_conditions)
    fixed_target_mask = keras.ops.concatenate(
        (
            keras.ops.ones(1),  # param 1 is inferred
            keras.ops.zeros(1),  # param 2 is fixed
        )
    )
    fixed_target_mask = np.broadcast_to(fixed_target_mask, (5, num_samples, 2)).reshape(-1, 2)
    targets_fixed = test_conditions_adapted["inference_variables"]
    targets_fixed = np.broadcast_to(np.expand_dims(targets_fixed, axis=1), (5, num_samples, 2)).reshape(-1, 2)

    fixed_samples = workflow.sample(
        num_samples=num_samples * 5, fixed_target_value=targets_fixed, fixed_target_mask=fixed_target_mask
    )["parameters"].reshape(5, num_samples, 2)
    assert (np.abs(fixed_samples[..., 1] - test_conditions["parameters"][:, 1:]) < 1e-6).all()
    assert (np.abs(fixed_samples[..., 0] - test_conditions["parameters"][:, :1]) > 0.1).any()  # should vary
