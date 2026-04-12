import keras
import numpy as np
import pytest


def test_sample_seed_determinism(inference_network):
    from bayesflow.networks import ScoringRuleNetwork

    xz_shape = (2, 3)
    conditions_shape = (2, 8)

    inference_network.build(xz_shape, conditions_shape=conditions_shape)

    batch_shape = (xz_shape[0],)

    if isinstance(inference_network, ScoringRuleNetwork):
        from bayesflow.scoring_rules import ParametricDistributionScore

        has_distribution = any(
            isinstance(score, ParametricDistributionScore) for score in inference_network.scoring_rules.values()
        )
        if not has_distribution:
            pytest.skip("This ScoringRuleNetwork has no parametric distribution scores to sample from")

    from bayesflow.networks import ConsistencyModel, StableConsistencyModel, DiffusionModel, FlowMatching

    sample_kwargs = dict(conditions=keras.random.normal(conditions_shape))
    # Pass steps=2 to iterative networks to keep the test fast
    if isinstance(inference_network, (ConsistencyModel, StableConsistencyModel, DiffusionModel, FlowMatching)):
        sample_kwargs["steps"] = 2

    samples_seed42_1 = inference_network.sample(batch_shape, seed=42, **sample_kwargs)
    samples_seed42_2 = inference_network.sample(batch_shape, seed=42, **sample_kwargs)
    samples_no_seed = inference_network.sample(batch_shape, **sample_kwargs)

    # deal with both return types dict and Tensor of sample methods (ScoringRuleNetwork returns dict)
    def leaves(x):
        return keras.tree.flatten(x)

    for s1, s2, s_unseeded in zip(leaves(samples_seed42_1), leaves(samples_seed42_2), leaves(samples_no_seed)):
        arr1 = keras.ops.convert_to_numpy(s1)
        arr2 = keras.ops.convert_to_numpy(s2)
        arr_unseeded = keras.ops.convert_to_numpy(s_unseeded)

        assert np.allclose(arr1, arr2), f"samples differ for identical seed ({inference_network})"
        assert not np.allclose(arr1, arr_unseeded), f"seeded and unseeded samples are identical ({inference_network})"
