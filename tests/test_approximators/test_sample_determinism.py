"""Seed handling for every approximator that supports ``.sample()``.

Sampling is reproducible for a given seed, independent across seeds, and never falls back to
Keras' global generator. The approximators covered here are the only ones defining ``sample()``;
``CompositionalApproximator`` and ``ModelComparisonApproximator`` inherit these code paths.
"""

import keras
import numpy as np
import pytest


@pytest.fixture()
def determinism_adapter():
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator.build_adapter(
        inference_variables=["mean", "std"],
        inference_conditions=["x"],
    )


@pytest.fixture()
def determinism_data():
    from tests.utils.normal_simulator import NormalSimulator

    return NormalSimulator().sample((4,))


def _continuous_approximator(adapter, inference_network):
    from bayesflow import ContinuousApproximator

    return ContinuousApproximator(adapter=adapter, inference_network=inference_network)


@pytest.fixture()
def coupling_flow_approximator(determinism_adapter):
    from bayesflow.networks import CouplingFlow

    return _continuous_approximator(determinism_adapter, CouplingFlow(depth=1, subnet_kwargs=dict(widths=(8, 8))))


@pytest.fixture()
def flow_matching_approximator(determinism_adapter):
    from bayesflow.networks import FlowMatching

    return _continuous_approximator(
        determinism_adapter,
        FlowMatching(subnet_kwargs=dict(widths=(8, 8)), integrate_kwargs=dict(steps=4)),
    )


@pytest.fixture()
def diffusion_model_approximator(determinism_adapter):
    from bayesflow.networks import DiffusionModel

    # euler_maruyama draws noise in every solver step, on top of the base distribution draw
    return _continuous_approximator(
        determinism_adapter,
        DiffusionModel(
            subnet_kwargs=dict(widths=(8, 8)),
            integrate_kwargs=dict(steps=4, method="euler_maruyama"),
        ),
    )


@pytest.fixture()
def consistency_model_approximator(determinism_adapter):
    from bayesflow.networks import ConsistencyModel

    return _continuous_approximator(
        determinism_adapter,
        ConsistencyModel(total_steps=10, subnet_kwargs=dict(widths=(8, 8))),
    )


@pytest.fixture()
def scoring_rule_approximator_for_determinism(determinism_adapter):
    from bayesflow import ScoringRuleApproximator
    from bayesflow.networks import ScoringRuleNetwork
    from bayesflow.scoring_rules import MvNormalScore, MixtureScore

    return ScoringRuleApproximator(
        adapter=determinism_adapter,
        inference_network=ScoringRuleNetwork(
            scoring_rules=dict(
                mvn=MvNormalScore(),
                mix=MixtureScore(mvn_c1=MvNormalScore(), mvn_c2=MvNormalScore()),
            ),
            subnet="mlp",
            subnet_kwargs=dict(widths=(8, 8)),
        ),
    )


@pytest.fixture()
def ensemble_approximator_for_determinism(determinism_adapter, coupling_flow_approximator):
    from bayesflow import EnsembleApproximator
    from bayesflow.networks import CouplingFlow

    other = _continuous_approximator(determinism_adapter, CouplingFlow(depth=1, subnet_kwargs=dict(widths=(8, 8))))
    return EnsembleApproximator(dict(member_1=coupling_flow_approximator, member_2=other))


@pytest.fixture(
    params=[
        "coupling_flow_approximator",
        "flow_matching_approximator",
        "diffusion_model_approximator",
        "consistency_model_approximator",
        "scoring_rule_approximator_for_determinism",
        "ensemble_approximator_for_determinism",
    ]
)
def sampling_approximator(request, determinism_adapter, determinism_data):
    approximator = request.getfixturevalue(request.param)

    batch = determinism_adapter(determinism_data)
    approximator.build(keras.tree.map_structure(keras.ops.shape, batch))

    return approximator


def assert_samples_equal(left, right, equal: bool, message: str):
    for key in left.keys():
        if equal:
            np.testing.assert_allclose(left[key], right[key], err_msg=f"{key}: {message}")
        else:
            with pytest.raises(AssertionError, match="Not equal to tolerance"):
                np.testing.assert_allclose(left[key], right[key], err_msg=f"{key}: {message}")


def test_sample_is_reproducible_for_equal_seeds(sampling_approximator, determinism_data):
    conditions = dict(x=determinism_data["x"])

    samples_1 = sampling_approximator.sample(num_samples=3, conditions=conditions, seed=42)
    samples_2 = sampling_approximator.sample(num_samples=3, conditions=conditions, seed=42)

    assert_samples_equal(samples_1, samples_2, equal=True, message="samples differ for identical seed")


def test_sample_differs_for_different_seeds(sampling_approximator, determinism_data):
    conditions = dict(x=determinism_data["x"])

    samples_1 = sampling_approximator.sample(num_samples=3, conditions=conditions, seed=42)
    samples_2 = sampling_approximator.sample(num_samples=3, conditions=conditions, seed=1337)

    assert_samples_equal(samples_1, samples_2, equal=False, message="samples identical for different seeds")


def test_integer_seed_matches_equivalent_seed_generator(sampling_approximator, determinism_data):
    conditions = dict(x=determinism_data["x"])

    from_integer = sampling_approximator.sample(num_samples=3, conditions=conditions, seed=42)
    from_generator = sampling_approximator.sample(
        num_samples=3, conditions=conditions, seed=keras.random.SeedGenerator(42)
    )

    assert_samples_equal(
        from_integer, from_generator, equal=True, message="integer seed and equivalent seed generator differ"
    )


def test_sample_advances_instance_seed_generator(sampling_approximator, determinism_data):
    """Without a seed, consecutive calls must draw from the advancing instance generator."""
    conditions = dict(x=determinism_data["x"])

    samples_1 = sampling_approximator.sample(num_samples=3, conditions=conditions)
    samples_2 = sampling_approximator.sample(num_samples=3, conditions=conditions)

    assert_samples_equal(samples_1, samples_2, equal=False, message="consecutive unseeded samples are identical")


def test_seed_generator_is_shared_across_condition_batches(sampling_approximator, determinism_data):
    """A single generator is shared across batches, instead of being re-seeded for each of them."""
    conditions = dict(x=determinism_data["x"])

    samples = sampling_approximator.sample(num_samples=3, conditions=conditions, batch_size=2, seed=42)

    for key, value in samples.items():
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            np.testing.assert_allclose(value[0:2], value[2:4], err_msg=f"{key}: condition batches repeat samples")


def test_sample_separate_is_seeded(scoring_rule_approximator_for_determinism, determinism_adapter, determinism_data):
    """`merge_scores=False` bypasses the mixture logic, but must still honor `seed`."""
    approximator = scoring_rule_approximator_for_determinism
    approximator.build(keras.tree.map_structure(keras.ops.shape, determinism_adapter(determinism_data)))

    conditions = dict(x=determinism_data["x"])

    samples_1 = approximator.sample(num_samples=3, conditions=conditions, merge_scores=False, seed=42)
    samples_2 = approximator.sample(num_samples=3, conditions=conditions, merge_scores=False, seed=42)
    samples_3 = approximator.sample(num_samples=3, conditions=conditions, merge_scores=False, seed=1337)

    for score_key in samples_1:
        assert_samples_equal(samples_1[score_key], samples_2[score_key], equal=True, message="differ for equal seed")
        assert_samples_equal(
            samples_1[score_key], samples_3[score_key], equal=False, message="identical for different seeds"
        )


@pytest.fixture()
def ancestral_approximator():
    from bayesflow import ContinuousApproximator
    from bayesflow.networks import CouplingFlow

    adapter = ContinuousApproximator.build_adapter(
        inference_variables=["beta"],
        inference_conditions=["mu", "x"],
    )
    approximator = ContinuousApproximator(
        adapter=adapter,
        inference_network=CouplingFlow(depth=1, subnet_kwargs=dict(widths=(8, 8))),
    )

    batch = adapter({key: np.random.standard_normal((4, 1)) for key in ("beta", "mu", "x")})
    approximator.build(keras.tree.map_structure(keras.ops.shape, batch))

    return approximator


@pytest.fixture()
def ancestral_conditions():
    """Four datasets with identical values, so repeated samples signal a re-seeded generator."""
    n_datasets, n_children, n_parent_samples = 4, 2, 2

    return (
        dict(x=np.tile(np.random.standard_normal((1, n_children, 1)), (n_datasets, 1, 1))),
        dict(mu=np.tile(np.random.standard_normal((1, n_parent_samples, 1)), (n_datasets, 1, 1))),
    )


def test_ancestral_sample_is_seeded(ancestral_approximator, ancestral_conditions):
    conditions, parent_conditions = ancestral_conditions

    kwargs = dict(conditions=conditions, ancestral_conditions=parent_conditions)
    samples_1 = ancestral_approximator.ancestral_sample(**kwargs, seed=42)
    samples_2 = ancestral_approximator.ancestral_sample(**kwargs, seed=42)
    samples_3 = ancestral_approximator.ancestral_sample(**kwargs, seed=1337)

    assert_samples_equal(samples_1, samples_2, equal=True, message="samples differ for identical seed")
    assert_samples_equal(samples_1, samples_3, equal=False, message="samples identical for different seeds")


def test_ancestral_sample_advances_instance_seed_generator(ancestral_approximator, ancestral_conditions):
    conditions, parent_conditions = ancestral_conditions

    kwargs = dict(conditions=conditions, ancestral_conditions=parent_conditions)
    samples_1 = ancestral_approximator.ancestral_sample(**kwargs)
    samples_2 = ancestral_approximator.ancestral_sample(**kwargs)

    assert_samples_equal(samples_1, samples_2, equal=False, message="consecutive unseeded samples are identical")


def test_ancestral_sample_shares_seed_generator_across_condition_batches(ancestral_approximator, ancestral_conditions):
    conditions, parent_conditions = ancestral_conditions

    samples = ancestral_approximator.ancestral_sample(
        conditions=conditions, ancestral_conditions=parent_conditions, batch_size=2, seed=42
    )

    for key, value in samples.items():
        with pytest.raises(AssertionError, match="Not equal to tolerance"):
            np.testing.assert_allclose(value[0:2], value[2:4], err_msg=f"{key}: condition batches repeat samples")


@pytest.mark.jax
def test_sample_is_traceable_under_jit(sampling_approximator, determinism_data):
    """An unseeded call must not reach Keras' global generator, which cannot be traced."""
    import jax

    conditions = dict(x=determinism_data["x"])

    # sample() itself is not jittable (numpy conversion), so trace the network call it wraps
    batch = sampling_approximator.adapter(dict(determinism_data) | conditions)
    inference_conditions = keras.ops.convert_to_tensor(batch["inference_conditions"])

    inference_network = getattr(sampling_approximator, "inference_network", None)
    if inference_network is None:
        pytest.skip("ensemble approximators have no single inference network")

    @jax.jit
    def sample_fn(inference_conditions):
        return inference_network.sample((keras.ops.shape(inference_conditions)[0],), conditions=inference_conditions)

    samples = sample_fn(inference_conditions)

    # scoring rule networks return one sample tensor per scoring rule
    batch_size = keras.ops.shape(inference_conditions)[0]
    for leaf in keras.tree.flatten(samples):
        assert keras.ops.shape(leaf)[0] == batch_size
