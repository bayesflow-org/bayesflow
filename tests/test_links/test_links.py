import keras
import numpy as np
import pytest


def test_link_output(link, generic_preactivation):
    output_shape = link.compute_output_shape(generic_preactivation.shape)
    output = link(generic_preactivation)

    assert output_shape == output.shape


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
            assert not np.all(
                np.diff(output, axis=i) > 0
            ), f"is ordered along axis which is not meant to be ordered: {i}."


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


def test_positive_semi_definite(random_matrix_batch):
    from bayesflow.links import PositiveSemiDefinite

    activation = PositiveSemiDefinite()

    output = activation(random_matrix_batch)

    output = keras.ops.convert_to_numpy(output)
    eigenvalues = np.linalg.eig(output).eigenvalues

    assert np.all(eigenvalues.real > 0) and np.all(
        np.isclose(eigenvalues.imag, 0)
    ), f"output is not positive semi-definite: real={eigenvalues.real}, imag={eigenvalues.imag}"
