import keras
import numpy as np

from bayesflow.masking import MaskHandler


def test_compositional_workflow_routes_semantic_masks():
    from bayesflow import CompositionalWorkflow
    from bayesflow.networks import DiffusionModel
    from bayesflow.simulators import TwoMoons

    num_samples = 3
    mask_handler = MaskHandler(
        fixed_target_prob=0.3,
        missing_target_prob=0.3,
        seed=41,
    )
    workflow = CompositionalWorkflow(
        inference_network=DiffusionModel(
            subnet="diffusion_transformer",
            subnet_kwargs={"widths": (8, 8)},
            integrate_kwargs={"method": "euler", "steps": 2},
            mask_handler=mask_handler,
        ),
        inference_variables=["parameters"],
        inference_conditions=["observables"],
        simulator=TwoMoons(),
    )

    workflow.fit_online(epochs=2, batch_size=2, num_batches_per_epoch=2)
    test_params = np.asarray(workflow.simulate(5)["parameters"], dtype=np.float32)
    simulator = TwoMoons()
    test_conditions = {
        "observables": np.asarray(
            [tuple(simulator.observation_model(t) for _ in range(3)) for t in test_params],
            dtype=np.float32,
        ),
        "parameters": test_params,
    }

    def prior_score_fn(theta):
        return {"parameters": keras.ops.zeros_like(theta["parameters"])}

    samples = workflow.compositional_sample(
        num_samples=num_samples,
        conditions=test_conditions,
        compute_prior_score=prior_score_fn,
    )["parameters"]

    fixed_target_mask = np.broadcast_to(
        np.asarray([1.0, 0.0], dtype=np.float32),
        (5, 2),
    ).copy()
    targets_fixed = workflow.adapter(test_conditions)["inference_variables"]
    fixed_samples = workflow.compositional_sample(
        conditions=test_conditions,
        num_samples=num_samples,
        compute_prior_score=prior_score_fn,
        fixed_target_value=targets_fixed,
        fixed_target_mask=fixed_target_mask,
    )["parameters"]

    assert samples.shape == fixed_samples.shape
    np.testing.assert_allclose(
        fixed_samples[..., 1],
        np.broadcast_to(test_conditions["parameters"][:, None, 1], (5, num_samples)),
        rtol=0.0,
        atol=1e-6,
    )
    assert np.any(np.abs(fixed_samples[..., 0] - test_conditions["parameters"][:, None, 0]) > 0.1)

    marginalized_samples = workflow.compositional_sample(
        conditions=test_conditions,
        num_samples=num_samples,
        compute_prior_score=prior_score_fn,
        infer_target_mask=fixed_target_mask,
    )["parameters"]
    assert samples.shape == marginalized_samples.shape
