```{toctree}
:hidden:

index
resources
```

# Getting Started with BayesFlow
BayesFlow is a Python library for Bayesian inference with neural networks. It supports tasks such as parameter estimation, model comparison, and model validation for both simulation-based models and traditional statistical models. It is especially useful when classical inference methods are unavailable, inefficient, or hard to apply.


## Install

We currently support Python 3.10 to 3.13. You can install the latest stable version from PyPI using:

```bash
pip install "bayesflow>=2.0"
```

If you want the latest features, you can install from source:

```bash
pip install git+https://github.com/bayesflow-org/bayesflow.git@dev
```

If you encounter problems with this or require more control, please refer to the instructions to install from source below.

### Backend

To use BayesFlow, you will also need to install one of the following machine learning backends.
Note that BayesFlow **will not run** without a backend.

- [Install JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [Install PyTorch](https://pytorch.org/get-started/locally/)
- [Install TensorFlow](https://www.tensorflow.org/install)

If you don't know which backend to use, we recommend JAX as it is currently the fastest backend.

As of version ``2.0.7``, the backend will be set automatically. If you have multiple backends, you can manually [set the backend environment variable as described by keras](https://keras.io/getting_started/#configuring-your-backend).
For example, inside your Python script write:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import bayesflow
```

If you use conda, you can alternatively set this individually for each environment in your terminal. For example:

```bash
conda env config vars set KERAS_BACKEND=jax
```

Or just plainly set the environment variable in your shell:

```bash
export KERAS_BACKEND=jax
```

## Getting Started

Using the high-level interface is easy, as demonstrated by the minimal working example below:

```python
import bayesflow as bf

workflow = bf.BasicWorkflow(
    inference_network=bf.networks.FlowMatching(),
    inference_variables=["parameters"],
    inference_conditions=["observables"],
    simulator=bf.simulators.SIR()
)

history = workflow.fit_online(epochs=20, batch_size=32, num_batches_per_epoch=200)

diagnostics = workflow.plot_default_diagnostics(test_data=300)
```

(getting-started-examples)=
# Examples

The tutorial notebooks are available in the repository-level `examples/` folder.


## Tutorial notebooks

1. [Bayesian Experimental Design](../_examples/Bayesian_Experimental_Design.ipynb) - Perform adaptive sequential experiments.
2. [Diffusion Models](../_examples/Diffusion_Models.ipynb) - A small tutorial on the power of diffusion models for SBI.
3. [Ensembles](../_examples/Ensembles.ipynb) - Train different networks at the same time and combine inferences.
4. [From ABC to BayesFlow](../_examples/From_ABC_to_BayesFlow.ipynb) - Upgrade from sequential to amortized inference.
5. [From BayesFlow 1.1 to 2.0](../_examples/From_BayesFlow_1.1_to_2.0.ipynb) - Learn how to migrate from BayesFlow 1.1 to 2.0.
6. [Likelihood Estimation](../_examples/Likelihood_Estimation.ipynb) - Learn synthetic likelihood functions.
7. [Linear Regression Starter](../_examples/Linear_Regression_Starter.ipynb) - Fit your first Bayesian regression with varying sample size.
8. [Lotka Volterra Point Estimation](../_examples/Lotka_Volterra_Point_Estimation.ipynb) - From simple point estimates to fully Bayesian inference.
9. [Multimodal Data](../_examples/Multimodal_Data.ipynb) - Fuse different data types for more informative inference.
10. [One Sample TTest](../_examples/One_Sample_TTest.ipynb) - Learn Bayes factors using probabilistic classification.
11. [Ratio Estimation](../_examples/Ratio_Estimation.ipynb) - Learn neural ratios for downstream MCMC sampling.
12. [SIR Posterior Estimation](../_examples/SIR_Posterior_Estimation.ipynb) - Model infectious diseases through an end-to-end Bayesian workflow.
13. [Spatial Data and Parameters](../_examples/Spatial_Data_and_Parameters.ipynb) - Learn parameters from or generate image data.
14. [Two Moons Starter](../_examples/Two_Moons_Starter.ipynb) - A simple starter example using the two moons benchmark.

### Books
Many examples from [Bayesian Cognitive Modeling: A Practical Course](https://bayesmodels.com/) by Lee & Wagenmakers (2013) in [BayesFlow](https://kucharssim.github.io/bayesflow-cognitive-modeling-book/).

## Reporting Issues

If you encounter any issues, please don't hesitate to open an issue on [Github](https://github.com/bayesflow-org/bayesflow/issues) or ask questions on our [Discourse Forums](https://discuss.bayesflow.org/).

## Getting Help

Please use the [BayesFlow Forums](https://discuss.bayesflow.org/) for any BayesFlow-related questions and discussions, and [GitHub Issues](https://github.com/bayesflow-org/bayesflow/issues) for bug reports and feature requests.


## Awesome Amortized Inference

If you are interested in a curated list of resources, including reviews, software, papers, and other resources related to amortized inference, feel free to explore our [community-driven list](https://github.com/bayesflow-org/awesome-amortized-inference). If you'd like a paper (by yourself or someone else) featured, please add it to the list with a pull request, an issue, or a message to the maintainers.


## License \& Source Code

BayesFlow is released under {mainbranch}`MIT License <LICENSE>`.
The source code is hosted on the public [GitHub repository](https://github.com/bayesflow-org/bayesflow).

Indices
-------

* {ref}`genindex`
* {ref}`modindex`

