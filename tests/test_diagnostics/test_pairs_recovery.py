import matplotlib.pyplot as plt
import numpy as np
import pytest

from bayesflow import diagnostics
from bayesflow.utils.exceptions import ShapeError


@pytest.fixture
def recovery_data():
    # Distinct, nonmonotonic columns and biased estimates expose swapped sources/axes.
    targets = np.array([[1, 100, -8], [4, 300, -2], [2, 200, -9], [8, 150, -4]], dtype=float)
    points = targets * np.array([2, 0.1, 3]) + np.array([4, 7, 2])
    points += np.array([[0, 3, -2], [2, -5, 1], [-1, 2, 4], [3, 1, -3]])
    draws = points[:, None, :] + np.array([-3, -1, 0, 2, 8])[None, :, None]
    return draws, targets


def test_cell_coordinates_and_geometry(recovery_data):
    draws, targets = recovery_data
    points = np.median(draws, axis=1)
    fig = diagnostics.pairs_recovery(draws, targets, uncertainty_agg=None, markersize=5)
    assert isinstance(fig, plt.Figure)
    assert diagnostics.pairs_recovery is diagnostics.plots.pairs_recovery
    axes = np.array(fig.axes).reshape(3, 3)
    fig.canvas.draw()
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            x = targets[:, j] if i >= j else points[:, j]
            y = targets[:, i] if i > j else points[:, i]
            np.testing.assert_allclose(ax.collections[0].get_offsets(), np.column_stack((x, y)))
            np.testing.assert_allclose(ax.collections[0].get_sizes(), [25])
            for label in (ax.xaxis.label, ax.yaxis.label):
                bounds = label.get_window_extent(fig.canvas.get_renderer())
                assert bounds.x0 >= 0 and bounds.y0 >= 0
                assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height
            assert ax.get_xlabel() == f"v_{j}\n({'Ground truth' if i >= j else 'Estimate'})"
            assert ax.get_ylabel() == f"v_{i}\n({'Ground truth' if i > j else 'Estimate'})"
            if i == j:
                np.testing.assert_allclose(ax.get_xlim(), ax.get_ylim())
                np.testing.assert_allclose(ax.lines[0].get_xdata(), ax.lines[0].get_ydata())
                assert ax.texts[0].get_text() == f"$r$ = {np.corrcoef(x, y)[0, 1]:.3f}"
                # Equal numeric limits must also be square on the rendered canvas.
                np.testing.assert_allclose(ax.bbox.width, ax.bbox.height)
            else:
                assert not ax.lines and not ax.texts
    # Changing a diagonal range must not rescale cells that show different data.
    old = axes[0, 1].get_ylim()
    axes[0, 0].set_ylim(-1000, 1000)
    assert axes[0, 1].get_ylim() == old
    assert not axes[0, 0].get_shared_x_axes().joined(axes[0, 0], axes[2, 0])


@pytest.mark.parametrize("kind", ["default", "credible", "symmetric", "bounds"])
def test_uncertainty_endpoints_and_aggregation(recovery_data, kind):
    draws, targets = recovery_data
    original = draws.copy()
    points = np.quantile(draws, 0.4, axis=1)
    kwargs = dict(point_agg=np.quantile, point_agg_kwargs={"q": 0.4}, add_corr=False)
    if kind == "default":
        bounds = np.quantile(draws, [0.025, 0.975], axis=1)
    elif kind == "credible":
        kwargs["uncertainty_agg_kwargs"] = {"prob": 0.5}
        bounds = np.quantile(draws, [0.25, 0.75], axis=1)
    elif kind == "symmetric":
        kwargs["uncertainty_agg"] = np.std
        kwargs["uncertainty_agg_kwargs"] = {"ddof": 1}
        spread = np.std(draws, axis=1, ddof=1)
        bounds = np.stack((points - spread, points + spread))
    else:
        bounds = np.stack((points - 20, points + 30))
        bounds.setflags(write=False)
        kwargs["uncertainty_agg"] = lambda x, axis: bounds
    before = bounds.copy()
    fig = diagnostics.pairs_recovery(draws, targets, **kwargs)
    axes = np.array(fig.axes).reshape(3, 3)
    np.testing.assert_allclose(axes[0, 2].collections[0].get_offsets(), points[:, [2, 0]])
    for i in range(3):
        ax = axes[i, i]
        container = ax.containers[0]
        np.testing.assert_allclose(container.lines[0].get_xdata(), targets[:, i])
        np.testing.assert_allclose(container.lines[0].get_ydata(), points[:, i])
        segments = container.lines[2][0].get_segments()
        expected = np.stack(
            (np.column_stack((targets[:, i], bounds[0, :, i])), np.column_stack((targets[:, i], bounds[1, :, i]))),
            axis=1,
        )
        np.testing.assert_allclose(segments, expected)
        assert ax.get_ylim()[0] <= bounds[0, :, i].min()
        assert ax.get_ylim()[1] >= bounds[1, :, i].max()
        assert not ax.texts
    np.testing.assert_array_equal(draws, original)
    np.testing.assert_array_equal(bounds, before)


def test_dictionary_selection_names_and_test_quantity(recovery_data):
    draws, truth = recovery_data
    estimates = {"beta": draws[..., :2], "sigma": draws[..., 2:]}
    targets = {"beta": truth[..., :2], "sigma": truth[..., 2:], "unused": np.ones((4, 7))}
    fig = diagnostics.pairs_recovery(estimates, targets, variable_keys="sigma", add_corr=False)
    assert len(fig.axes) == 1
    assert fig.axes[0].get_xlabel() == "sigma\n(Ground truth)"
    fig = diagnostics.pairs_recovery(
        estimates,
        targets,
        variable_keys=("sigma", "beta"),
        variable_names=("s", "b0", "b1"),
        test_quantities={"sum": lambda data: data["beta"].sum(axis=-1)},
        uncertainty_agg=None,
    )
    axes = np.array(fig.axes).reshape(4, 4)
    expected_draws = np.concatenate(
        (draws[..., :2].sum(axis=-1, keepdims=True), draws[..., 2:], draws[..., :2]), axis=-1
    )
    expected_truth = np.column_stack((truth[..., :2].sum(axis=-1), truth[..., 2:], truth[..., :2]))
    expected_points = np.median(expected_draws, axis=1)
    for i, name in enumerate(["sum", "s", "b0", "b1"]):
        assert axes[i, i].get_xlabel() == f"{name}\n(Ground truth)"
        np.testing.assert_allclose(
            axes[i, i].collections[0].get_offsets(), np.column_stack((expected_truth[:, i], expected_points[:, i]))
        )
    assert list(estimates) == ["beta", "sigma"]
    assert list(targets) == ["beta", "sigma", "unused"]


@pytest.mark.parametrize(
    "draw_shape,truth_shape",
    [
        ((4, 3), (4, 3)),
        ((4, 5, 3), (3, 3)),
        ((4, 5, 3), (4, 2)),
        ((4, 0, 3), (4, 3)),
        ((0, 5, 3), (0, 3)),
        ((4, 5, 0), (4, 0)),
    ],
)
def test_invalid_shapes_do_not_create_figure(draw_shape, truth_shape):
    before = plt.get_fignums()
    # Variable-name validation in dicts_to_arrays precedes shape validation.
    error = ValueError if truth_shape[-1] != draw_shape[-1] else ShapeError
    with pytest.raises(error):
        diagnostics.pairs_recovery(np.ones(draw_shape), np.ones(truth_shape))
    assert plt.get_fignums() == before


def test_invalid_aggregate_shapes(recovery_data):
    draws, targets = recovery_data
    with pytest.raises(ShapeError, match="point_agg"):
        diagnostics.pairs_recovery(draws, targets, point_agg=lambda x, axis: x.mean())
    with pytest.raises(ShapeError, match="uncertainty_agg"):
        diagnostics.pairs_recovery(draws, targets, uncertainty_agg=lambda x, axis: np.ones((4, 1)))


def test_array_test_quantities_rejected(recovery_data):
    with pytest.raises(TypeError, match="dictionaries"):
        diagnostics.pairs_recovery(*recovery_data, test_quantities={"sum": lambda data: data.sum(axis=-1)})


def test_inferred_keys_and_names_with_test_quantity(recovery_data):
    draws, targets = recovery_data
    fig = diagnostics.pairs_recovery(
        {"sigma": draws[..., 2:]},
        {"sigma": targets[..., 2:]},
        test_quantities={"square": lambda data: data["sigma"][:, 0] ** 2},
        uncertainty_agg=None,
    )
    assert fig.axes[0].get_xlabel() == "square\n(Ground truth)"
    assert fig.axes[-1].get_xlabel() == "sigma\n(Ground truth)"


@pytest.mark.parametrize("bounds", [False, True])
def test_invalid_uncertainty_does_not_create_figure(recovery_data, bounds):
    draws, targets = recovery_data
    invalid = np.ones((2, *targets.shape)) * 1000 if bounds else -np.ones(targets.shape)
    before = plt.get_fignums()
    with pytest.raises(ValueError, match="nonnegative"):
        diagnostics.pairs_recovery(draws, targets, uncertainty_agg=lambda x, axis: invalid)
    assert plt.get_fignums() == before


def test_array_labels_and_single_dataset(recovery_data):
    draws, targets = recovery_data
    fig = diagnostics.pairs_recovery(
        draws[:1, :, :1],
        targets[:1, :1],
        variable_names="custom",
        add_corr=False,
        height=4,
    )
    np.testing.assert_allclose(fig.get_size_inches(), [4, 4])
    assert fig.axes[0].get_xlabel() == "custom\n(Ground truth)"
    assert fig.axes[0].get_ylabel() == "custom\n(Estimate)"


def test_string_selection_with_test_quantity(recovery_data):
    draws, targets = recovery_data
    fig = diagnostics.pairs_recovery(
        {"sigma": draws[..., 2:]},
        {"sigma": targets[..., 2:]},
        variable_keys="sigma",
        variable_names="s",
        test_quantities={"square": lambda data: data["sigma"][:, 0] ** 2},
        uncertainty_agg=None,
    )
    assert len(fig.axes) == 4
    np.testing.assert_allclose(
        fig.axes[0].collections[0].get_offsets(),
        np.column_stack((targets[:, 2] ** 2, np.median(draws[:, :, 2] ** 2, axis=1))),
    )
    assert fig.axes[-1].get_xlabel() == "s\n(Ground truth)"
