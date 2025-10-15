import keras
import numpy as np
import pytest


def test_invalid_shape_for_ordered_quantiles(ordered_quantiles, batch_size, num_quantiles, num_variables):
    with pytest.raises(AssertionError) as excinfo:
        ordered_quantiles.build((batch_size, batch_size, num_quantiles, num_variables))

    assert "resolve which axis should be ordered automatically" in str(excinfo)


@pytest.mark.parametrize("axis", [1, 2])
def test_invalid_shape_for_ordered_quantiles_with_specified_axis(
    ordered_quantiles, axis, batch_size, num_quantiles, num_variables
):
    ordered_quantiles.axis = axis
    ordered_quantiles.build((batch_size, batch_size, num_quantiles, num_variables))


def check_ordering(output, axis):
    output = keras.ops.convert_to_numpy(output)
    assert np.all(np.diff(output, axis=axis) > 0), f"is not ordered along specified axis: {axis}."
    for i in range(output.ndim):
        if i != axis % output.ndim:
            assert not np.all(np.diff(output, axis=i) > 0), (
                f"is ordered along axis which is not meant to be ordered: {i}."
            )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_ordering(axis, unordered):
    from bayesflow.links import Ordered

    activation = Ordered(axis=axis, anchor_index=5)

    output = activation(unordered)

    check_ordering(output, axis)


def test_quantile_ordering(quantiles, unordered):
    from bayesflow.links import OrderedQuantiles

    activation = OrderedQuantiles(q=quantiles)

    activation.build(unordered.shape)
    axis = activation.axis

    output = activation(unordered)

    check_ordering(output, axis)


def test_cholesky_factor(cholesky_factor, batch_size, num_variables):
    input_shape = cholesky_factor.compute_input_shape((batch_size, num_variables, num_variables))

    random_preactivation = keras.random.normal(input_shape)

    output = cholesky_factor(random_preactivation)
    output = keras.ops.convert_to_numpy(output)

    np.testing.assert_allclose(
        np.triu(output, k=1),
        np.zeros((batch_size, num_variables, num_variables)),
        atol=1e-4,
        err_msg=f"All elements above diagonal must be zero for lower triangular matrix: {output}",
    )

    diag = np.diagonal(output, axis1=1, axis2=2)
    assert np.all(diag > 0), f"diagonal is not strictly positive: {diag}"
