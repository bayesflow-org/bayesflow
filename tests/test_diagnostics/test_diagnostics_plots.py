import bayesflow as bf
import numpy as np
import pytest


def num_variables(x: dict):
    return sum(arr.shape[-1] for arr in x.values())


def test_backend():
    import matplotlib.pyplot as plt

    # if the local testing backend is not Agg
    # then you may run into issues once you run workflow tests
    # on GitHub, since these use the Agg backend
    assert plt.get_backend() == "Agg"


def test_calibration_ecdf(random_estimates, random_targets, var_names):
    print(random_estimates, random_targets, var_names)

    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.calibration_ecdf(random_estimates, random_targets)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "beta_1"

    # custom variable names
    out = bf.diagnostics.plots.calibration_ecdf(
        estimates=random_estimates,
        targets=random_targets,
        variable_names=var_names,
    )
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "$\\beta_1$"

    # subset of keys with a single scalar key
    out = bf.diagnostics.plots.calibration_ecdf(
        estimates=random_estimates, targets=random_targets, variable_keys="sigma"
    )
    assert len(out.axes) == random_estimates["sigma"].shape[-1]
    assert out.axes[0].title._text == "sigma"

    # use single array instead of dict of arrays as input
    out = bf.diagnostics.plots.calibration_ecdf(
        estimates=random_estimates["beta"],
        targets=random_targets["beta"],
    )
    assert len(out.axes) == random_estimates["beta"].shape[-1]
    # cannot infer the variable names from an array so default names are used
    assert out.axes[1].title._text == "v_1"

    # test quantities plots are shown
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.plots.calibration_ecdf(random_estimates, random_targets, test_quantities=test_quantities)
    assert len(out.axes) == len(test_quantities) + num_variables(random_estimates)
    assert out.axes[1].title._text == r"$\beta_1 \cdot \beta_2$"
    assert out.axes[-1].title._text == r"sigma"

    # test plot titles changed to variable_names in case test quantities exist
    out = bf.diagnostics.plots.calibration_ecdf(
        random_estimates, random_targets, test_quantities=test_quantities, variable_names=var_names
    )
    assert out.axes[-1].title._text == r"$\sigma$"


def test_calibration_ecdf_from_quantiles(random_estimates, random_targets, var_names):
    quantile_levels = [0.1, 0.5, 0.9]

    estimates = {
        variable_name: {"quantiles": np.moveaxis(np.quantile(value, q=quantile_levels, axis=1), 0, 1)}
        for variable_name, value in random_estimates.items()
    }

    out = bf.diagnostics.calibration_ecdf_from_quantiles(estimates, random_targets, quantile_levels=quantile_levels)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "beta_1"


def test_calibration_histogram(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.calibration_histogram(random_estimates, random_targets)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[0].title._text == "beta_0"

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.plots.calibration_histogram(random_estimates, random_targets, test_quantities=test_quantities)
    assert len(out.axes) == len(test_quantities) + num_variables(random_estimates)
    assert out.axes[1].title._text == r"$\beta_1 \cdot \beta_2$"
    assert out.axes[-1].title._text == r"sigma"


def test_loss(history):
    out = bf.diagnostics.loss(history)
    assert len(out.axes) == 1
    assert out.axes[0].title._text == "Loss Trajectory"


def test_recovery_bounds(random_estimates, random_targets):
    # basic functionality: automatic variable names
    from bayesflow.utils.numpy_utils import credible_interval

    out = bf.diagnostics.plots.recovery(
        random_estimates, random_targets, markersize=4, uncertainty_agg=credible_interval
    )
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[2].title._text == "sigma"

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.plots.calibration_histogram(random_estimates, random_targets, test_quantities=test_quantities)
    assert len(out.axes) == len(test_quantities) + num_variables(random_estimates)
    assert out.axes[1].title._text == r"$\beta_1 \cdot \beta_2$"
    assert out.axes[-1].title._text == r"sigma"


def test_recovery_symmetric(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.recovery(random_estimates, random_targets, markersize=4, uncertainty_agg=np.std)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[2].title._text == "sigma"


def test_recovery_from_estimates(random_estimates, random_targets):
    # basic functionality: automatic variable names
    estimates = {variable_name: {"mean": np.mean(value, axis=1)} for variable_name, value in random_estimates.items()}

    out = bf.diagnostics.plots.recovery_from_estimates(
        estimates, random_targets, markersize=4, marker_mapping={"mean": "x"}
    )
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[2].title._text == "sigma"


def test_z_score_contraction(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.z_score_contraction(random_estimates, random_targets, markersize=4)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "beta_1"

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.plots.z_score_contraction(random_estimates, random_targets, test_quantities=test_quantities)
    assert len(out.axes) == len(test_quantities) + num_variables(random_estimates)
    assert out.axes[1].title._text == r"$\beta_1 \cdot \beta_2$"
    assert out.axes[-1].title._text == r"sigma"


def test_pairs_samples(random_priors):
    out = bf.diagnostics.plots.pairs_samples(
        samples=random_priors,
        variable_keys=["beta", "sigma"],
        markersize=4,
    )
    num_vars = random_priors["sigma"].shape[-1] + random_priors["beta"].shape[-1]
    assert out.axes.shape == (num_vars, num_vars)
    assert out.axes[0, 0].get_ylabel() == "beta_0"
    assert out.axes[2, 2].get_xlabel() == "sigma"


def test_pairs_posterior(random_estimates, random_targets, random_priors):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.pairs_posterior(
        random_estimates, random_targets, dataset_id=1, markersize=4, target_markersize=4
    )
    num_vars = num_variables(random_estimates)
    assert out.axes.shape == (num_vars, num_vars)
    assert out.axes[0, 0].get_ylabel() == "beta_0"
    assert out.axes[2, 2].get_xlabel() == "sigma"

    # also plot priors
    out = bf.diagnostics.plots.pairs_posterior(
        estimates=random_estimates,
        targets=random_targets,
        priors=random_priors,
        dataset_id=1,
    )
    num_vars = num_variables(random_estimates)
    assert out.axes.shape == (num_vars, num_vars)
    assert out.axes[0, 0].get_ylabel() == "beta_0"
    assert out.axes[2, 2].get_xlabel() == "sigma"
    assert out.figure.legends[0].get_texts()[0]._text == "Prior"

    # Test legend placement toggle
    out = bf.diagnostics.plots.pairs_posterior(
        estimates=random_estimates,
        targets=random_targets,
        dataset_id=1,
        place_legend_below=True,
    )
    assert len(out.figure.legends) >= 1

    # Placement should change location, number of columns, and anchor point
    legends = out.figure.legends[0]
    assert legends._loc == "upper center" or legends._loc == 9
    assert legends._ncols == 3

    bbox = legends.get_bbox_to_anchor()
    anchor = getattr(bbox, "_bbox", bbox)
    assert np.isclose(anchor.x0, 0.5)
    assert np.isclose(anchor.y0, 0.0)

    with pytest.raises(ValueError):
        bf.diagnostics.plots.pairs_posterior(
            estimates=random_estimates,
            targets=random_targets,
            priors=random_priors,
            dataset_id=[1, 3],
        )


def test_pairs_quantity(random_estimates, random_targets, random_priors):
    # test test_quantities and label assignment
    key = next(iter(random_estimates.keys()))
    test_quantities = {
        "a": lambda data: np.sum(data[key], axis=-1),
        "b": lambda data: np.prod(data[key], axis=-1),
    }
    out = bf.diagnostics.plots.pairs_quantity(
        values=bf.diagnostics.posterior_contraction,
        estimates=random_estimates,
        targets=random_targets,
        test_quantities=test_quantities,
    )

    num_vars = num_variables(random_estimates) + len(test_quantities)
    assert out.axes.shape == (num_vars, num_vars)
    assert out.axes[0, 0].get_ylabel() == "a"
    assert out.axes[2, 0].get_ylabel() == "beta_0"
    assert out.axes[4, 4].get_xlabel() == "sigma"

    values = bf.diagnostics.posterior_contraction(estimates=random_estimates, targets=random_targets, aggregation=None)

    bf.diagnostics.plots.pairs_quantity(
        values,
        targets=random_targets,
    )

    raw_values = np.random.normal(size=values["values"].shape)
    out = bf.diagnostics.plots.pairs_quantity(raw_values, targets=random_targets, variable_keys=["beta", "sigma"])
    assert out.axes.shape == (3, 3)

    with pytest.raises(ValueError):
        bf.diagnostics.plots.pairs_quantity(raw_values, targets=random_targets)

    with pytest.raises(ValueError):
        bf.diagnostics.plots.pairs_quantity(
            values=values,
            estimates=random_estimates,
            targets=random_targets,
            test_quantities=test_quantities,
        )

    with pytest.raises(ValueError):
        bf.diagnostics.plots.pairs_quantity(
            values=bf.diagnostics.posterior_contraction,
            targets=random_targets,
        )


def test_plot_quantity(random_estimates, random_targets, random_priors):
    # test test_quantities and label assignment
    key = next(iter(random_estimates.keys()))
    test_quantities = {
        "a": lambda data: np.sum(data[key], axis=-1),
        "b": lambda data: np.prod(data[key], axis=-1),
    }
    out = bf.diagnostics.plots.plot_quantity(
        values=bf.diagnostics.posterior_contraction,
        estimates=random_estimates,
        targets=random_targets,
        test_quantities=test_quantities,
    )

    num_vars = num_variables(random_estimates) + len(test_quantities)
    assert len(out.axes) == num_vars
    assert out.axes[0].title._text == "a"

    values = bf.diagnostics.posterior_contraction(estimates=random_estimates, targets=random_targets, aggregation=None)

    bf.diagnostics.plots.plot_quantity(
        values,
        targets=random_targets,
    )

    raw_values = np.random.normal(size=values["values"].shape)
    out = bf.diagnostics.plots.plot_quantity(raw_values, targets=random_targets, variable_keys=["beta", "sigma"])
    assert len(out.axes) == 3

    with pytest.raises(ValueError):
        bf.diagnostics.plots.plot_quantity(raw_values, targets=random_targets)

    with pytest.raises(ValueError):
        bf.diagnostics.plots.plot_quantity(
            values=values,
            estimates=random_estimates,
            targets=random_targets,
            test_quantities=test_quantities,
        )

    with pytest.raises(ValueError):
        bf.diagnostics.plots.plot_quantity(
            values=bf.diagnostics.posterior_contraction,
            targets=random_targets,
        )


def test_mc_calibration(pred_models, true_models, model_names):
    out = bf.diagnostics.plots.mc_calibration(pred_models, true_models, model_names=model_names, markersize=4)
    assert len(out.axes) == pred_models.shape[-1]
    assert out.axes[0].get_ylabel() == "True Probability"
    assert out.axes[0].get_xlabel() == "Predicted Probability"
    assert out.axes[-1].get_title() == r"$\mathcal{M}_2$"


def test_mc_confusion_matrix(pred_models, true_models, model_names):
    out = bf.diagnostics.plots.mc_confusion_matrix(pred_models, true_models, model_names, normalize="true")
    assert out.axes[0].get_ylabel() == "True model"
    assert out.axes[0].get_xlabel() == "Predicted model"
    assert out.axes[0].get_title() == "Confusion Matrix"


def test_mc_confusion_matrix_into_axis(pred_models, true_models, model_names):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = bf.diagnostics.plots.mc_confusion_matrix(pred_models, true_models, model_names, ax=ax)
    # Draws into the provided axis and returns its parent figure (enables composition).
    assert out is fig
    assert ax.get_ylabel() == "True model"
    assert ax.get_xlabel() == "Predicted model"
    assert ax.get_title() == "Confusion Matrix"
    plt.close(fig)


def test_coverage(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.coverage(random_estimates, random_targets)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "beta_1"
    assert out.axes[0].get_xlabel() == "Central interval width"
    assert out.axes[0].get_ylabel() == "Empirical coverage"

    # test quantities
    test_quantities = {
        r"$\beta_1 + \beta_2$": lambda data: np.sum(data["beta"], axis=-1),
        r"$\beta_1 \cdot \beta_2$": lambda data: np.prod(data["beta"], axis=-1),
    }
    out = bf.diagnostics.plots.coverage(random_estimates, random_targets, test_quantities=test_quantities)
    assert len(out.axes) == len(test_quantities) + num_variables(random_estimates)
    assert out.axes[1].title._text == r"$\beta_1 \cdot \beta_2$"
    assert out.axes[-1].title._text == r"sigma"


def test_coverage_diff(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.coverage(random_estimates, random_targets, difference=True)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "beta_1"
    assert out.axes[0].get_xlabel() == "Central interval width"
    assert out.axes[0].get_ylabel() == "Empirical coverage difference"


def test_bayes_factor_recovery(pred_log_bayes_factors, true_log_bayes_factors, true_models, model_names):
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models,
        model_names=model_names,
    )
    num_competing = pred_log_bayes_factors.shape[-1]
    assert len(out.axes) == num_competing
    assert "log" in out.axes[0].get_xlabel().lower()
    assert "log" in out.axes[0].get_ylabel().lower()


def test_pairwise_bayes_factors(pred_log_bayes_factors, true_models, model_names):
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_models,
        model_names=model_names,
    )
    assert out.axes is not None


def test_pairwise_bayes_factors_into_axis(pred_log_bayes_factors, true_models, model_names):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_models,
        model_names=model_names,
        ax=ax,
    )
    # Draws into the provided axis and returns its parent figure (enables composition).
    assert out is fig
    assert ax.get_ylabel() == "True model"
    assert ax.get_title() == "Mean log Bayes factor"
    plt.close(fig)


def test_pairwise_bayes_factors_blue_only_when_embedded(pred_log_bayes_factors, true_models, model_names):
    """The brighter accent blue appears only when embedded in an axis; standalone stays navy."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # Standalone (ax=None) -> navy blue end.
    standalone = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors, true_models=true_models, model_names=model_names
    )
    standalone_blue = mcolors.to_hex(standalone.axes[0].images[0].cmap(1.0))
    assert standalone_blue == "#132a70"
    plt.close(standalone)

    # Embedded (ax provided) -> brighter Bayes factor accent blue.
    fig, ax = plt.subplots()
    bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors, true_models=true_models, model_names=model_names, ax=ax
    )
    embedded_blue = mcolors.to_hex(ax.images[0].cmap(1.0))
    assert embedded_blue == "#6969ff"
    plt.close(fig)


def test_confusion_matrix_and_pairwise_bayes_factors_side_by_side(
    pred_models, pred_log_bayes_factors, true_models, model_names
):
    """The confusion matrix and pairwise Bayes factor heatmap compose into a shared figure."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    cm_fig = bf.diagnostics.plots.mc_confusion_matrix(pred_models, true_models, model_names, ax=axes[0])
    pbf_fig = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_models,
        model_names=model_names,
        ax=axes[1],
    )
    # Both draw into the same shared figure...
    assert cm_fig is fig and pbf_fig is fig
    # ...each panel plus its own colorbar -> 4 axes total...
    assert len(fig.axes) == 4
    # ...and each panel keeps its own title (no grand overarching title).
    assert axes[0].get_title() == "Confusion Matrix"
    assert axes[1].get_title() == "Mean log Bayes factor"
    # Embedded pairwise heatmap uses the brighter accent blue to stand apart from the navy matrix.
    import matplotlib.colors as mcolors

    assert mcolors.to_hex(axes[1].images[0].cmap(1.0)) == "#6969ff"
    plt.close(fig)


def test_pairwise_bayes_factors_integer_labels(pred_log_bayes_factors, true_models):
    true_model_idx = np.argmax(true_models, axis=-1)
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_model_idx,
    )
    assert out.axes is not None


def test_pairwise_bayes_factors_no_title(pred_log_bayes_factors, true_models):
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_models,
        title=False,
    )
    assert out.axes[0].get_title() == ""


def test_pairwise_bayes_factors_custom_fmt(pred_log_bayes_factors, true_models):
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_models,
        fmt=".2f",
    )
    assert out.axes is not None


def test_pairwise_bayes_factors_custom_cmap(pred_log_bayes_factors, true_models):
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_models=true_models,
        cmap="coolwarm",
    )
    assert out.axes is not None


def test_pairwise_bayes_factors_zero_matrix():
    # all-zero mean matrix triggers abs_max = 0 → fallback to 1.0
    pred = np.zeros((10, 2))
    true_models = np.eye(3)[np.zeros(10, dtype=int)]
    out = bf.diagnostics.plots.pairwise_bayes_factors(
        pred_log_bayes_factors=pred,
        true_models=true_models,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_shape_mismatch():
    pred = np.random.normal(size=(10, 2))
    true_wrong = np.random.normal(size=(10, 3))
    true_models = np.eye(3)[np.random.choice(3, 10)]
    with pytest.raises(ValueError, match="same shape"):
        bf.diagnostics.plots.bayes_factor_recovery(
            pred_log_bayes_factors=pred,
            true_log_bayes_factors=true_wrong,
            true_models=true_models,
        )


def test_bayes_factor_recovery_integer_labels(pred_log_bayes_factors, true_log_bayes_factors, true_models):
    true_model_idx = np.argmax(true_models, axis=-1)
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_model_idx,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_no_corr(pred_log_bayes_factors, true_log_bayes_factors, true_models):
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models,
        add_corr=False,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_markersize(pred_log_bayes_factors, true_log_bayes_factors, true_models):
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models,
        markersize=5.0,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_custom_layout(pred_log_bayes_factors, true_log_bayes_factors, true_models):
    # Use a 2×2 grid (more panels than needed) to exercise custom num_col/num_row
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models,
        num_col=2,
        num_row=2,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_only_num_row(pred_log_bayes_factors, true_log_bayes_factors, true_models):
    # num_row set, num_col inferred → covers elif num_col is None branch
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models,
        num_row=1,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_only_num_col(pred_log_bayes_factors, true_log_bayes_factors, true_models):
    # num_col set, num_row inferred → covers elif num_row is None branch
    # num_col=1 also exercises the single-column layout path in add_y_labels / add_x_labels
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models,
        num_col=1,
    )
    assert out.axes is not None


def test_bayes_factor_recovery_missing_model(pred_log_bayes_factors, true_log_bayes_factors):
    # All samples from model 0 → models 1 and 2 have empty masks → covers continue branch
    true_models_skewed = np.zeros((100, 3), dtype=np.int32)
    true_models_skewed[:, 0] = 1
    out = bf.diagnostics.plots.bayes_factor_recovery(
        pred_log_bayes_factors=pred_log_bayes_factors,
        true_log_bayes_factors=true_log_bayes_factors,
        true_models=true_models_skewed,
    )
    assert out.axes is not None
