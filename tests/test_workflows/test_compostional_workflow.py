import keras
import numpy as np
import pytest


def test_compositional_sampling():
    from bayesflow.networks import DiffusionModel, TimeSeriesNetwork
    from bayesflow import CompositionalWorkflow
    from bayesflow.simulators import SIR

    num_samples = 3
    batch_size = 2
    num_batches_per_epoch = 2
    epochs = 5
    workflow = CompositionalWorkflow(
        inference_network=DiffusionModel(
            subnet_kwargs=dict(widths=(8, 8)),
            drop_target_prob=0.5,
        ),
        summary_network=TimeSeriesNetwork(),
        inference_variables=["parameters"],
        summary_variables=["observables"],
        simulator=SIR(subsample=None),
    )

    workflow.fit_online(epochs=epochs, batch_size=batch_size, num_batches_per_epoch=num_batches_per_epoch)
    test_params = workflow.simulate(5)["parameters"]
    test_conditions = {
        "observables": np.array(
            [(SIR().observation_model(t), SIR().observation_model(t), SIR().observation_model(t)) for t in test_params]
        )
    }
    test_conditions.update({"parameters": test_params})

    def prior_score_fn(theta, time):
        # placeholder prior score
        return {"parameters": (1 - time) * keras.ops.zeros(keras.ops.shape(theta["parameters"]))}

    samples = workflow.compositional_sample(
        num_samples=num_samples, conditions=test_conditions, compute_prior_score=prior_score_fn, return_summaries=True
    )
    assert samples["parameters"].shape == (5, num_samples, 2)

    # use precomputed summaries
    samples = workflow.compositional_sample(
        num_samples=num_samples, compute_prior_score=prior_score_fn, summaries=samples["_summaries"]
    )
    assert samples["parameters"].shape == (5, num_samples, 2)


# ---- Guidance (slower) ----------------------------------------


def test_compositional_masking():
    from bayesflow.networks import DiffusionModel
    from bayesflow import CompositionalWorkflow
    from bayesflow.simulators import TwoMoons

    num_samples = 3
    batch_size = 2
    num_batches_per_epoch = 2
    epochs = 2
    workflow = CompositionalWorkflow(
        inference_network=DiffusionModel(
            subnet_kwargs=dict(widths=(8, 8)),
            drop_target_prob=0.5,
        ),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=epochs, batch_size=batch_size, num_batches_per_epoch=num_batches_per_epoch)
    test_params = workflow.simulate(5)["parameters"]
    test_conditions = {
        "observables": np.array(
            [
                (TwoMoons().observation_model(t), TwoMoons().observation_model(t), TwoMoons().observation_model(t))
                for t in test_params
            ]
        )
    }
    test_conditions.update({"parameters": test_params})

    def prior_score_fn(theta, time):
        # uniform prior (should be transformed to unbounded prior for a real application)
        return {"parameters": (1 - time) * keras.ops.zeros(keras.ops.shape(theta["parameters"]))}

    samples = workflow.compositional_sample(
        num_samples=num_samples, conditions=test_conditions, compute_prior_score=prior_score_fn
    )["parameters"]

    test_conditions_adapted = workflow.adapter(test_conditions)
    target_mask = keras.ops.concatenate(
        (
            keras.ops.ones(1),  # param 1 is inferred
            keras.ops.zeros(1),  # param 2 is fixed
        )
    )
    targets_fixed = test_conditions_adapted["inference_variables"][0]  # one set of parameters
    if "inference_variables" in workflow.approximator.standardize_layers:
        targets_fixed = workflow.approximator.standardize_layers["inference_variables"](targets_fixed, forward=True)

    fixed_samples = workflow.compositional_sample(
        conditions=test_conditions,
        num_samples=num_samples,
        compute_prior_score=prior_score_fn,
        targets_fixed=targets_fixed,
        target_mask=target_mask,
    )["parameters"]
    assert samples.shape == fixed_samples.shape
    assert (np.abs(fixed_samples[..., 1] - test_conditions["parameters"][0, 1]) < 1e-6).all()
    assert (np.abs(fixed_samples[..., 0] - test_conditions["parameters"][0, 0]) > 0.1).any()  # should vary


@pytest.mark.slow
def test_diffusion_compositional_guidance():
    from bayesflow import CompositionalWorkflow
    from bayesflow.simulators import TwoMoons
    from bayesflow.networks import DiffusionModel

    workflow = CompositionalWorkflow(
        inference_network=DiffusionModel(
            subnet_kwargs={"widths": (32, 32)},
        ),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=2, batch_size=2, num_batches_per_epoch=2, verbose=0)
    test_params = workflow.simulate(5)["parameters"]
    test_conditions = {
        "observables": np.array(
            [
                (TwoMoons().observation_model(t), TwoMoons().observation_model(t), TwoMoons().observation_model(t))
                for t in test_params
            ]
        )
    }

    def prior_score_fn(theta, time):
        # uniform prior (should be transformed to unbounded prior for a real application)
        return {"parameters": (1 - time) * keras.ops.zeros(keras.ops.shape(theta["parameters"]))}

    samples = workflow.compositional_sample(
        num_samples=2, conditions=test_conditions, compute_prior_score=prior_score_fn
    )["parameters"]

    def constraint(z):
        params = workflow.approximator.standardize_layers["inference_variables"](z, forward=False)
        a1 = params[..., 0]
        return a1

    samples_guided = workflow.compositional_sample(
        num_samples=2,
        conditions=test_conditions,
        compute_prior_score=prior_score_fn,
        guidance_constraints=dict(constraints=constraint),
    )["parameters"]
    assert samples_guided.shape == samples.shape
    assert (samples_guided[..., 0] < 0).all()

    def guidance_function(x, time):
        return x * 0

    samples_guided_func = workflow.compositional_sample(
        num_samples=2,
        conditions=test_conditions,
        compute_prior_score=prior_score_fn,
        guidance_function=guidance_function,
    )["parameters"]
    assert samples_guided_func.shape == samples.shape


def test_compositional_workflow_from_basic():
    from bayesflow import BasicWorkflow, CompositionalWorkflow
    from bayesflow.networks import DiffusionModel
    from bayesflow.simulators import TwoMoons

    sim = TwoMoons()

    basic_wf = BasicWorkflow(
        inference_network=DiffusionModel(subnet_kwargs=dict(widths=(8, 8))),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=sim,
    )
    basic_wf.fit_online(epochs=2, batch_size=4, num_batches_per_epoch=3, verbose=0)

    comp_wf = CompositionalWorkflow.from_basic_workflow(basic_wf)

    assert isinstance(comp_wf, CompositionalWorkflow)

    # inference network is a copy — independent objects, same weights
    assert comp_wf.approximator.inference_network is not basic_wf.approximator.inference_network
    src_weights = basic_wf.approximator.inference_network.get_weights()
    cpy_weights = comp_wf.approximator.inference_network.get_weights()
    assert all(np.array_equal(a, b) for a, b in zip(src_weights, cpy_weights))

    # standardizer is a copy — independent objects, same learned statistics
    assert comp_wf.approximator.standardizer is not basic_wf.approximator.standardizer
    for var in basic_wf.approximator.standardizer.standardize_layers:
        src_layer = basic_wf.approximator.standardizer.standardize_layers[var]
        cpy_layer = comp_wf.approximator.standardizer.standardize_layers[var]
        for s, c in zip(src_layer.get_weights(), cpy_layer.get_weights()):
            assert np.array_equal(s, c), f"standardizer weights for '{var}' differ"

    for var, layer in comp_wf.approximator.standardizer.standardize_layers.items():
        for count in layer.count:
            assert float(count) > 0, f"count for '{var}' should be non-zero after training"

    assert comp_wf.simulator is basic_wf.simulator

    # the compositional workflow can produce samples of the correct shape
    test_data = sim.sample(5)
    conditions = {
        "observables": np.stack([test_data["observables"]] * 2, axis=1),
        "parameters": test_data["parameters"],
    }

    def zero_prior_score(theta, time):
        return {"parameters": (1 - time) * keras.ops.zeros(keras.ops.shape(theta["parameters"]))}

    samples = comp_wf.compositional_sample(num_samples=3, conditions=conditions, compute_prior_score=zero_prior_score)
    assert samples["parameters"].shape == (5, 3, 2)
