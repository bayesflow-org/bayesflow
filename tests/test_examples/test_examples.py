import pytest

from tests.utils import run_notebook


@pytest.mark.skip(reason="requires setting up Stan")
@pytest.mark.slow
def test_bayesian_experimental_design(examples_path):
    run_notebook(examples_path / "Bayesian_Experimental_Design.ipynb")


@pytest.mark.skip(reason="requires setting up pyabc")
@pytest.mark.slow
def test_from_abc_to_bayesflow(examples_path):
    run_notebook(examples_path / "From_ABC_to_BayesFlow.ipynb")


@pytest.mark.slow
def test_linear_regression_starter(examples_path):
    run_notebook(examples_path / "Linear_Regression_Starter.ipynb")


@pytest.mark.slow
def test_lotka_volterra_point_estimation_and_expert_stats(examples_path):
    run_notebook(examples_path / "Lotka_Volterra_Point_Estimation.ipynb")


@pytest.mark.slow
def test_one_sample_ttest(examples_path):
    run_notebook(examples_path / "One_Sample_TTest.ipynb")


@pytest.mark.slow
def test_sir_posterior_estimation(examples_path):
    run_notebook(examples_path / "SIR_Posterior_Estimation.ipynb")


@pytest.mark.slow
def test_two_moons_starter(examples_path):
    run_notebook(examples_path / "Two_Moons_Starter.ipynb")


@pytest.mark.slow
def test_likelihood_estimation(examples_path):
    run_notebook(examples_path / "Likelihood_Estimation.ipynb")
