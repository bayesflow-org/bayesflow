import math
import keras
import numpy as np
import pytest

from bayesflow.networks import CouplingFlow
from bayesflow.networks.inference.coupling.permutations import FixedPermutation
from bayesflow.utils import jacobian, log_abs_det
from tests.utils import assert_allclose

TOL = 1e-4


def make_mask(batch_size, xz_dim):
    mask = np.ones((batch_size, xz_dim), dtype="float32")
    mask[0, 1] = mask[1, 0] = mask[1, -1] = 0
    return mask


def build_with_random_weights(flow, random_samples, random_conditions):
    """Perturb network weights; keras inits with an identity map, which would hide masking bugs."""
    conditions_shape = None if random_conditions is None else keras.ops.shape(random_conditions)
    flow.build(keras.ops.shape(random_samples), conditions_shape)
    for i, weight in enumerate(flow.trainable_weights):
        weight.assign(weight + keras.random.normal(weight.shape, stddev=0.3, seed=i))
    return flow


def permute_like_flow(flow, xz):
    """Permute the dims (e.g. of a mask) like the flow's permutation layers do."""
    xz = keras.ops.convert_to_numpy(xz)
    for layer in flow.invertible_layers:
        if isinstance(layer, FixedPermutation):
            idx = keras.ops.convert_to_numpy(layer.forward_indices)
            xz = xz[..., idx]
    return xz


@pytest.fixture()
def flow(coupling_flow, random_samples, random_conditions):
    return build_with_random_weights(coupling_flow, random_samples, random_conditions)


def test_mask_of_ones_is_moot(flow, random_samples, random_conditions):
    shape = keras.ops.shape(random_samples)[-1:]
    ones = np.ones(shape, dtype="float32")
    z, log_density = flow(random_samples, conditions=random_conditions, density=True)
    z_masked, log_density_masked = flow(
        random_samples, conditions=random_conditions, density=True, fixed_target_mask=ones
    )

    np.testing.assert_array_equal(keras.ops.convert_to_numpy(z), keras.ops.convert_to_numpy(z_masked))
    assert_allclose(log_density, log_density_masked, rtol=TOL, atol=TOL)


def test_fixed_dims_are_unchanged(flow, random_samples, random_conditions):
    mask = make_mask(*keras.ops.shape(random_samples))
    z = flow(random_samples, conditions=random_conditions, fixed_target_mask=mask)
    fixed = permute_like_flow(flow, mask) == 0
    z_fixed = keras.ops.convert_to_numpy(z)[fixed]
    random_samples_fixed = permute_like_flow(flow, random_samples)[fixed]

    np.testing.assert_array_equal(z_fixed, random_samples_fixed)


def test_density_matches_numerical_jac(flow, random_samples, random_conditions):
    """Fixed dims use the identity map; the full log_abs_det should be the same as of the unmasked block."""
    mask = make_mask(*keras.ops.shape(random_samples))
    _, log_density = flow(random_samples, conditions=random_conditions, density=True, fixed_target_mask=mask)

    def _forward(x):
        return flow(x, conditions=random_conditions, fixed_target_mask=mask)

    z, numerical_jacobian = jacobian(_forward, random_samples, return_output=True)
    not_masked = permute_like_flow(flow, mask) == 1
    z = keras.ops.convert_to_numpy(z)
    base_log_prob = np.where(not_masked, -0.5 * z**2 - 0.5 * math.log(2 * math.pi), 0.0)
    base_log_prob = np.sum(base_log_prob, axis=-1)
    numerical_log_det = keras.ops.convert_to_numpy(log_abs_det(numerical_jacobian))
    numerical_log_density = base_log_prob + numerical_log_det

    assert_allclose(log_density, numerical_log_density, rtol=TOL, atol=TOL)


def test_inverse_undoes_forward(flow, random_samples, random_conditions):
    mask = make_mask(*keras.ops.shape(random_samples))
    kwargs = {"conditions": random_conditions, "density": True, "fixed_target_mask": mask}

    z, forward_log_density = flow(random_samples, **kwargs)
    x, inverse_log_density = flow(z, inverse=True, fixed_target_value=random_samples, **kwargs)

    assert_allclose(x, random_samples, rtol=TOL, atol=TOL)
    assert_allclose(inverse_log_density, forward_log_density, rtol=TOL, atol=TOL)


def test_samples_keep_fixed_values(flow, random_samples, random_conditions):
    batch_size = keras.ops.shape(random_samples)[0]
    mask = make_mask(*keras.ops.shape(random_samples))

    inputs_fixed = keras.ops.convert_to_numpy(random_samples)[mask == 0]
    outputs = flow.sample(
        batch_size, conditions=random_conditions, fixed_target_mask=mask, fixed_target_value=random_samples
    )
    outputs_fixed = keras.ops.convert_to_numpy(outputs)[mask == 0]

    np.testing.assert_array_equal(inputs_fixed, outputs_fixed)


def test_orthogonal_permutation_raises(random_samples, random_conditions):
    """Submatrices of the learned orthogonal matrix can be singular, so masking is unsafe here."""
    flow = CouplingFlow(depth=1, permutation="orthogonal", subnet_kwargs={"widths": [8]})
    flow = build_with_random_weights(flow, random_samples, random_conditions)
    mask = make_mask(*keras.ops.shape(random_samples))

    with pytest.raises(ValueError, match="orthogonal"):
        flow.log_prob(random_samples, random_conditions, fixed_target_mask=mask)


def test_infer_target_mask_raises(flow, random_samples, random_conditions):
    """A coupling flow has no tractable *marginal* density, so infer_target should not be allowed."""
    mask = make_mask(*keras.ops.shape(random_samples))

    with pytest.raises(ValueError, match="infer_target_mask"):
        flow.log_prob(random_samples, random_conditions, infer_target_mask=mask)
