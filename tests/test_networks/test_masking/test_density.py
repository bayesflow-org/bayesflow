import keras
import numpy as np
import pytest

from bayesflow.adapters import Adapter
from bayesflow.approximators import ContinuousApproximator
from bayesflow.distributions import DiagonalNormal, DiagonalStudentT, Mixture
from bayesflow.networks.helpers.standardization.standardize import Standardize

from .conftest import make_network


def _numpy(value):
    return keras.ops.convert_to_numpy(value)


class ZeroSubnet(keras.Layer):
    def call(
        self,
        inputs,
        training=None,
        fixed_target_mask=None,
        infer_target_mask=None,
        observed_condition_mask=None,
        **kwargs,
    ):
        return keras.ops.zeros_like(inputs[0])

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class HandlerlessDensityNetwork(keras.Layer):
    def build(self, input_shape, conditions_shape=None):
        super().build(input_shape)

    def call(self, inputs):
        return inputs

    def log_prob(self, samples, conditions=None, **kwargs):
        return -keras.ops.sum(keras.ops.square(samples), axis=-1)


def test_diagonal_normal_mask_is_a_normalized_marginal(targets, fixed_target_mask):
    distribution = DiagonalNormal()
    distribution.build(tuple(targets.shape))

    actual = distribution.log_prob(targets, mask=fixed_target_mask)
    elementwise = -np.float32(0.5) * (_numpy(targets) ** np.float32(2.0) + np.float32(1.8378770664))
    expected = np.sum(elementwise * _numpy(fixed_target_mask), axis=-1, dtype=np.float32)

    np.testing.assert_allclose(_numpy(actual), expected, rtol=1e-6, atol=1e-6)


def test_diagonal_normal_all_one_and_all_zero_promises(targets):
    distribution = DiagonalNormal()
    distribution.build(tuple(targets.shape))

    unmasked = distribution.log_prob(targets)
    all_one = distribution.log_prob(targets, mask=keras.ops.ones_like(targets))
    all_zero = distribution.log_prob(targets, mask=keras.ops.zeros_like(targets))

    np.testing.assert_allclose(_numpy(all_one), _numpy(unmasked), rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(_numpy(all_zero), np.zeros((2,), dtype=np.float32))


def test_mixture_propagates_density_mask(targets, fixed_target_mask):
    components = [DiagonalNormal(mean=-1.0), DiagonalNormal(mean=1.0)]
    mixture = Mixture(components)
    mixture.build(tuple(targets.shape))

    actual = mixture.log_prob(targets, mask=fixed_target_mask)
    component_log_probs = keras.ops.stack(
        [component.log_prob(targets, mask=fixed_target_mask) for component in components],
        axis=-1,
    )
    expected = keras.ops.logsumexp(
        component_log_probs + keras.ops.log_softmax(mixture._mixture_logits),
        axis=-1,
    )

    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=1e-6, atol=1e-6)


def test_diagonal_student_t_mask_is_exact_lower_dimensional_marginal(targets, fixed_target_mask):
    loc = np.asarray([-1.0, 0.5, 2.0], dtype=np.float32)
    scale = np.asarray([0.5, 2.0, 1.5], dtype=np.float32)
    distribution = DiagonalStudentT(df=7, loc=loc, scale=scale)
    distribution.build(tuple(targets.shape))
    marginal = DiagonalStudentT(df=7, loc=loc[[0, 2]], scale=scale[[0, 2]])
    marginal.build((targets.shape[0], 2))

    actual = distribution.log_prob(targets, mask=fixed_target_mask)
    expected = marginal.log_prob(keras.ops.take(targets, [0, 2], axis=-1))

    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=1e-6, atol=1e-6)


def test_diagonal_student_t_mask_isolates_nan_coordinates(targets, fixed_target_mask):
    values = _numpy(targets).astype(np.float32)
    values[:, 1] = np.nan
    values = keras.ops.convert_to_tensor(values)
    distribution = DiagonalStudentT(df=7)
    distribution.build(tuple(values.shape))
    marginal = DiagonalStudentT(df=7)
    marginal.build((values.shape[0], 2))

    actual = distribution.log_prob(values, mask=fixed_target_mask)
    expected = marginal.log_prob(keras.ops.take(values, [0, 2], axis=-1))

    assert np.all(np.isfinite(_numpy(actual)))
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=1e-6, atol=1e-6)


def test_diagonal_student_t_empty_marginal_has_unit_density(targets):
    distribution = DiagonalStudentT(df=7)
    distribution.build(tuple(targets.shape))

    actual = distribution.log_prob(targets, mask=keras.ops.zeros_like(targets))

    np.testing.assert_array_equal(_numpy(actual), np.zeros((2,), dtype=np.float32))


@pytest.mark.parametrize("network_name", ["diffusion", "flow_matching"])
def test_network_density_ignores_all_fixed_coordinates(network_name, targets, conditions):
    network = make_network(network_name)
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))

    log_prob = network.log_prob(
        targets,
        conditions=conditions,
        fixed_target_mask=keras.ops.zeros_like(targets),
    )

    np.testing.assert_allclose(
        _numpy(log_prob),
        np.zeros((2,), dtype=np.float32),
        rtol=0.0,
        atol=1e-5,
    )


@pytest.mark.parametrize("network_name", ["diffusion", "flow_matching"])
def test_all_one_density_mask_is_moot_and_needs_no_fixed_value(network_name, targets, conditions):
    network = make_network(network_name)
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))

    unmasked = network.log_prob(targets, conditions=conditions)
    masked = network.log_prob(
        targets,
        conditions=conditions,
        fixed_target_mask=keras.ops.ones_like(targets),
    )

    np.testing.assert_allclose(_numpy(masked), _numpy(unmasked), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mask_name", ["fixed_target_mask", "infer_target_mask"])
@pytest.mark.parametrize("network_name", ["diffusion", "flow_matching"])
def test_partial_density_excludes_fixed_coordinate_for_coordinatewise_field(
    network_name,
    mask_name,
    targets,
    conditions,
    fixed_target_mask,
):
    network = make_network(network_name, subnet=ZeroSubnet())
    network.build(tuple(targets.shape), conditions_shape=tuple(conditions.shape))
    changed = keras.ops.convert_to_tensor(_numpy(targets).copy())
    changed_values = _numpy(changed)
    changed_values[:, 1] += np.float32(100.0)
    changed = keras.ops.convert_to_tensor(changed_values)

    original_log_prob = network.log_prob(
        targets,
        conditions=conditions,
        **{mask_name: fixed_target_mask},
    )
    changed_log_prob = network.log_prob(
        changed,
        conditions=conditions,
        **{mask_name: fixed_target_mask},
    )

    np.testing.assert_allclose(
        _numpy(changed_log_prob),
        _numpy(original_log_prob),
        rtol=1e-5,
        atol=1e-5,
    )


def test_approximator_masks_dynamic_standardizer_jacobian(targets, conditions, fixed_target_mask):
    network = make_network("flow_matching")
    approximator = ContinuousApproximator(
        inference_network=network,
        adapter=Adapter(),
        standardize="inference_variables",
    )
    approximator.build(
        {
            "inference_variables": tuple(targets.shape),
            "inference_conditions": tuple(conditions.shape),
        }
    )
    approximator.compute_metrics(
        targets,
        inference_conditions=conditions,
        stage="training",
    )

    data = {
        "inference_variables": _numpy(targets).astype(np.float32),
        "inference_conditions": _numpy(conditions).astype(np.float32),
        "fixed_target_mask": _numpy(fixed_target_mask).astype(np.float32),
    }
    actual = approximator.log_prob(data, to_numpy=False)

    target_values = _numpy(targets).astype(np.float32)
    mean = np.mean(target_values, axis=0, dtype=np.float32)
    variance = np.mean((target_values - mean) ** np.float32(2.0), axis=0, dtype=np.float32)
    std = np.sqrt(variance).astype(np.float32)
    standardized = keras.ops.convert_to_tensor(((target_values - mean) / std).astype(np.float32))
    log_det = -np.sum(
        np.log(std).astype(np.float32) * _numpy(fixed_target_mask),
        axis=-1,
        dtype=np.float32,
    )
    expected = network.log_prob(
        standardized,
        conditions=conditions,
        fixed_target_mask=fixed_target_mask,
    ) + keras.ops.convert_to_tensor(log_det)

    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=1e-5, atol=1e-5)


def test_handlerless_density_ignores_semantic_masks(targets, fixed_target_mask):
    approximator = ContinuousApproximator(
        inference_network=HandlerlessDensityNetwork(),
        adapter=Adapter(),
        standardize="inference_variables",
    )
    approximator.build({"inference_variables": tuple(targets.shape)})
    approximator.standardizer.maybe_standardize(targets, key="inference_variables", stage="training")

    data = {"inference_variables": _numpy(targets).astype(np.float32)}
    unmasked = approximator.log_prob(data, to_numpy=False)
    masked = approximator.log_prob(
        data | {"fixed_target_mask": _numpy(fixed_target_mask).astype(np.float32)},
        to_numpy=False,
    )

    np.testing.assert_allclose(_numpy(masked), _numpy(unmasked), rtol=0.0, atol=1e-6)


def test_standardize_density_mask_broadcasts_and_selects_jacobian_coordinates():
    layer = Standardize()
    layer.build((2, 3))
    std = np.asarray([2.0, 4.0, 0.5], dtype=np.float32)
    layer.moving_m2[0].assign(std ** np.float32(2.0))
    layer.count[0].assign(np.float32(1.0))
    values = keras.ops.convert_to_tensor(np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))

    masks = {
        "partial": np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        "all": np.ones((3,), dtype=np.float32),
        "none": np.zeros((3,), dtype=np.float32),
    }
    for mask in masks.values():
        _, forward = layer(
            values,
            stage="inference",
            forward=True,
            log_det_jac=True,
            log_det_jac_mask=keras.ops.convert_to_tensor(mask),
        )
        _, inverse = layer(
            values,
            stage="inference",
            forward=False,
            log_det_jac=True,
            log_det_jac_mask=keras.ops.convert_to_tensor(mask),
        )
        expected = np.full(
            (2,),
            -np.sum(np.log(std) * mask, dtype=np.float32),
            dtype=np.float32,
        )

        np.testing.assert_allclose(_numpy(forward), expected, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(_numpy(inverse), -expected, rtol=0.0, atol=1e-6)


def test_standardize_nested_density_masks_follow_each_leaf():
    layer = Standardize()
    layer.build({"a": (2, 2), "b": (2, 3)})
    std_a = np.asarray([2.0, 4.0], dtype=np.float32)
    std_b = np.asarray([0.5, 2.0, 8.0], dtype=np.float32)
    for index, std in enumerate((std_a, std_b)):
        layer.moving_m2[index].assign(std ** np.float32(2.0))
        layer.count[index].assign(np.float32(1.0))

    values = {
        "a": keras.ops.ones((2, 2), dtype="float32"),
        "b": keras.ops.ones((2, 3), dtype="float32"),
    }
    masks = {
        "a": keras.ops.convert_to_tensor(np.asarray([1.0, 0.0], dtype=np.float32)),
        "b": keras.ops.convert_to_tensor(np.asarray([0.0, 1.0, 1.0], dtype=np.float32)),
    }

    _, actual = layer(
        values,
        stage="inference",
        forward=True,
        log_det_jac=True,
        log_det_jac_mask=masks,
    )

    expected_a = np.full((2,), -np.log(std_a[0]), dtype=np.float32)
    expected_b = np.full((2,), -np.sum(np.log(std_b[1:]), dtype=np.float32), dtype=np.float32)
    np.testing.assert_allclose(_numpy(actual["a"]), expected_a, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(_numpy(actual["b"]), expected_b, rtol=0.0, atol=1e-6)
