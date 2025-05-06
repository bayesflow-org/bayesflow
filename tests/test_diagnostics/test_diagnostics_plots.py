import bayesflow as bf
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


def test_calibration_histogram(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.calibration_histogram(random_estimates, random_targets)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[0].title._text == "beta_0"


def test_loss(history):
    out = bf.diagnostics.loss(history)
    assert len(out.axes) == 1
    assert out.axes[0].title._text == "Loss Trajectory"


def test_recovery(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.recovery(random_estimates, random_targets)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[2].title._text == "sigma"


def test_z_score_contraction(random_estimates, random_targets):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.z_score_contraction(random_estimates, random_targets)
    assert len(out.axes) == num_variables(random_estimates)
    assert out.axes[1].title._text == "beta_1"


def test_pairs_samples(random_priors):
    out = bf.diagnostics.plots.pairs_samples(
        samples=random_priors,
        variable_keys=["beta", "sigma"],
    )
    num_vars = random_priors["sigma"].shape[-1] + random_priors["beta"].shape[-1]
    assert out.axes.shape == (num_vars, num_vars)
    assert out.axes[0, 0].get_ylabel() == "beta_0"
    assert out.axes[2, 2].get_xlabel() == "sigma"


def test_pairs_posterior(random_estimates, random_targets, random_priors):
    # basic functionality: automatic variable names
    out = bf.diagnostics.plots.pairs_posterior(
        random_estimates,
        random_targets,
        dataset_id=1,
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

    with pytest.raises(ValueError):
        bf.diagnostics.plots.pairs_posterior(
            estimates=random_estimates,
            targets=random_targets,
            priors=random_priors,
            dataset_id=[1, 3],
        )


def test_mc_calibration(pred_models, true_models, model_names):
    out = bf.diagnostics.plots.mc_calibration(pred_models, true_models, model_names=model_names)
    assert len(out.axes) == pred_models.shape[-1]
    assert out.axes[0].get_ylabel() == "True Probability"
    assert out.axes[0].get_xlabel() == "Predicted Probability"
    assert out.axes[-1].get_title() == r"$\mathcal{M}_2$"


def test_mc_confusion_matrix(pred_models, true_models, model_names):
    out = bf.diagnostics.plots.mc_confusion_matrix(pred_models, true_models, model_names, normalize="true")
    assert out.axes[0].get_ylabel() == "True model"
    assert out.axes[0].get_xlabel() == "Predicted model"
    assert out.axes[0].get_title() == "Confusion Matrix"
