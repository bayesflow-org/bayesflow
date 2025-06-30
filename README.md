# BayesFlow <img src="img/bayesflow_hex.png" style="float: right; width: 20%; height: 20%;" align="right" alt="BayesFlow Logo" />
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/bayesflow-org/bayesflow/tests.yaml?style=for-the-badge&label=Tests)
![Codecov](https://img.shields.io/codecov/c/github/bayesflow-org/bayesflow?style=for-the-badge&link=https%3A%2F%2Fapp.codecov.io%2Fgh%2Fbayesflow-org%2Fbayesflow%2Ftree%2Fmain)
[![DOI](https://img.shields.io/badge/DOI-10.21105%2Fjoss.05702-blue?style=for-the-badge)](https://doi.org/10.21105/joss.05702)
![PyPI - License](https://img.shields.io/pypi/l/bayesflow?style=for-the-badge)
![NumFOCUS Affiliated Project](https://img.shields.io/badge/NumFOCUS-Affiliated%20Project-orange?style=for-the-badge)

BayesFlow is a Python library for simulation-based **Amortized Bayesian Inference** with neural networks.
It provides users and researchers with:

- A user-friendly API for rapid Bayesian workflows
- A rich collection of neural network architectures
- Multi-backend support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)

BayesFlow (version 2+) is designed to be a flexible and efficient tool that enables rapid statistical inference
fueled by continuous progress in generative AI and Bayesian inference.

> [!IMPORTANT]
> As the 2.0 version introduced many new features, we still have to make breaking changes from time to time.
> This especially concerns **saving and loading** of models. We aim to stabilize this from the 2.1 release onwards.
> Until then, consider pinning your BayesFlow 2.0 installation to an exact version, or re-training after an update
> for less costly models.

## Important Note for Existing Users

You are currently looking at BayesFlow 2.0+, which is a complete rewrite of the library.
While it shares the same overall goals with the 1.x versions, the API is not compatible.

> [!CAUTION]
> A few features, most notably hierarchical models, have not been ported to BayesFlow 2.0+
> yet. We are working on those features and plan to add them soon. You can find the complete
> list in the [FAQ](#faq) below.

The [Moving from BayesFlow v1.1 to v2.0](examples/From_BayesFlow_1.1_to_2.0.ipynb) guide
highlights how concepts and classes relate between the two versions.

## Conceptual Overview

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./img/bayesflow_landing_dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./img/bayesflow_landing_light.jpg">
  <img alt="Overview graphic on using BayesFlow. It is split in three columns: 1. Choose your backend: BayesFlow is based on Keras, so you can choose PyTorch, TensorFlow or JAX. 2. Define your simulator: You specify your simulator in Python, and use it to generate simulated data. 3. Choose your algorithm: You define a generative neural network that you can use for estimation after training." src="./img/bayesflow_landing_dark.jpg">
</picture>
</div>

A cornerstone idea of amortized Bayesian inference is to employ generative
neural networks for parameter estimation, model comparison, and model validation
when working with intractable simulators whose behavior as a whole is too
complex to be described analytically.

## Install

We currently support Python 3.10 to 3.12. You can install the latest stable version from PyPI using:

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

Once installed, [set the backend environment variable as required by keras](https://keras.io/getting_started/#configuring-your-backend).
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

This way, you also don't have to manually set the backend every time you are starting Python to use BayesFlow.

## Getting Started

Using the high-level interface is easy, as demonstrated by the minimal working example below:

```python
import bayesflow as bf

workflow = bf.BasicWorkflow(
    inference_network=bf.networks.CouplingFlow(),
    summary_network=bf.networks.TimeSeriesNetwork(),
    inference_variables=["parameters"],
    summary_variables=["observables"],
    simulator=bf.simulators.SIR()
)

history = workflow.fit_online(epochs=15, batch_size=32, num_batches_per_epoch=200)

diagnostics = workflow.plot_default_diagnostics(test_data=300)
```

For an in-depth exposition, check out our expanding list of resources below.

### Books

Many examples from [Bayesian Cognitive Modeling: A Practical Course](https://bayesmodels.com/) by Lee & Wagenmakers (2013) in [BayesFlow](https://kucharssim.github.io/bayesflow-cognitive-modeling-book/).

### Tutorial notebooks

1. [Linear regression starter example](examples/Linear_Regression_Starter.ipynb)
2. [From ABC to BayesFlow](examples/From_ABC_to_BayesFlow.ipynb)
3. [Two moons starter example](examples/Two_Moons_Starter.ipynb)
4. [Rapid iteration with point estimators](examples/Lotka_Volterra_Point_Estimation.ipynb)
5. [SIR model with custom summary network](examples/SIR_Posterior_Estimation.ipynb)
6. [Bayesian experimental design](examples/Bayesian_Experimental_Design.ipynb)
7. [Simple model comparison example](examples/One_Sample_TTest.ipynb)
8. [Likelihood estimation](examples/Likelihood_Estimation.ipynb)
9. [Moving from BayesFlow v1.1 to v2.0](examples/From_BayesFlow_1.1_to_2.0.ipynb)

More tutorials are always welcome! Please consider making a pull request if you have a cool application that you want to contribute.

## Contributing

If you want to contribute to BayesFlow, we recommend installing it from source, see [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## Reporting Issues

If you encounter any issues, please don't hesitate to open an issue here on [Github](https://github.com/bayesflow-org/bayesflow/issues) or ask questions on our [Discourse Forums](https://discuss.bayesflow.org/).

## Documentation \& Help

Documentation is available at https://bayesflow.org. Please use the [BayesFlow Forums](https://discuss.bayesflow.org/) for any BayesFlow-related questions and discussions, and [GitHub Issues](https://github.com/bayesflow-org/bayesflow/issues) for bug reports and feature requests.

## Citing BayesFlow

You can cite BayesFlow along the lines of:

- We approximated the posterior using neural posterior estimation (NPE) with learned summary statistics (Radev et al., 2020), as implemented in the BayesFlow framework for amortized Bayesian inference (Radev et al., 2023a).
- We approximated the likelihood using neural likelihood estimation (NLE) without hand-crafted summary statistics (Papamakarios et al., 2019), leveraging its implementation in BayesFlow for efficient and flexible inference.

1. Radev, S. T., Schmitt, M., Schumacher, L., Elsemüller, L., Pratz, V., Schälte, Y., Köthe, U., & Bürkner, P.-C. (2023a). BayesFlow: Amortized Bayesian workflows with neural networks. *The Journal of Open Source Software, 8(89)*, 5702.([arXiv](https://arxiv.org/abs/2306.16015))([JOSS](https://joss.theoj.org/papers/10.21105/joss.05702))
2. Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., Köthe, U. (2020). BayesFlow: Learning complex stochastic models with invertible neural networks. *IEEE Transactions on Neural Networks and Learning Systems, 33(4)*, 1452-1466. ([arXiv](https://arxiv.org/abs/2003.06281))([IEEE TNNLS](https://ieeexplore.ieee.org/document/9298920))
3. Radev, S. T., Schmitt, M., Pratz, V., Picchini, U., Köthe, U., & Bürkner, P.-C. (2023b). JANA: Jointly amortized neural approximation of complex Bayesian models. *Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence, 216*, 1695-1706. ([arXiv](https://arxiv.org/abs/2302.09125))([PMLR](https://proceedings.mlr.press/v216/radev23a.html))

**BibTeX:**

```
@article{bayesflow_2023_software,
  title = {{BayesFlow}: Amortized {B}ayesian workflows with neural networks},
  author = {Radev, Stefan T. and Schmitt, Marvin and Schumacher, Lukas and Elsemüller, Lasse and Pratz, Valentin and Schälte, Yannik and Köthe, Ullrich and Bürkner, Paul-Christian},
  journal = {Journal of Open Source Software},
  volume = {8},
  number = {89},
  pages = {5702},
  year = {2023}
}

@article{bayesflow_2020_original,
  title = {{BayesFlow}: Learning complex stochastic models with invertible neural networks},
  author = {Radev, Stefan T. and Mertens, Ulf K. and Voss, Andreas and Ardizzone, Lynton and K{\"o}the, Ullrich},
  journal = {IEEE transactions on neural networks and learning systems},
  volume = {33},
  number = {4},
  pages = {1452--1466},
  year = {2020}
}

@inproceedings{bayesflow_2023_jana,
  title = {{JANA}: Jointly amortized neural approximation of complex {B}ayesian models},
  author = {Radev, Stefan T. and Schmitt, Marvin and Pratz, Valentin and Picchini, Umberto and K\"othe, Ullrich and B\"urkner, Paul-Christian},
  booktitle = {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = {1695--1706},
  year = {2023},
  volume = {216},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

## FAQ

-------------

**Question:**
I am starting with Bayesflow, which backend should I use?

**Answer:**
We recommend JAX as it is currently the fastest backend.

-------------

**Question:**
I am getting `ModuleNotFoundError: No module named 'tensorflow'` when I try to import BayesFlow.

**Answer:**
One of these applies:
- You want to use tensorflow as your backend, but you have not installed it.
See [here](https://www.tensorflow.org/install).


- You want to use a backend other than tensorflow, but have not set the environment variable correctly.
See [here](https://keras.io/getting_started/#configuring-your-backend).


- You have set the environment variable, but it is not being picked up by Python.
This can happen silently in some development environments (e.g., VSCode or PyCharm).
Try setting the backend as shown [here](https://keras.io/getting_started/#configuring-your-backend)
in your Python script via `os.environ`.

-------------

**Question:**
What is the difference between Bayesflow 2.0+ and previous versions?

**Answer:**
BayesFlow 2.0+ is a complete rewrite of the library. It shares the same
overall goals with previous versions, but has much better modularity
and extensibility. What is more, the new BayesFlow has multi-backend support via Keras3,
while the old version was based on TensorFlow.

-------------

**Question:**
Should I switch to BayesFlow 2.0+ now? Are there features that are still missing?

**Answer:**
In general, we recommend to switch, as the new version is easier to use and will continue
to receive improvements and new features. However, a few features are still missing, so you
might want to wait until everything you need has been ported to BayesFlow 2.0+.

Depending on your needs, you might not want to upgrade yet if one of the following applies:

- You have an ongoing project that uses BayesFlow 1.x, and you do not want to allocate
  time for migrating it to the new API.
- You have already trained models in BayesFlow 1.x, that you do not want to re-train
  with the new version. Loading models from version 1.x in version 2.0+ is not supported.
- You require a feature that was not ported to BayesFlow 2.0+ yet. To our knowledge,
  this applies to:
  * Two-level/Hierarchical models (planned for version 2.1): `TwoLevelGenerativeModel`, `TwoLevelPrior`.
  * Sensitivity analysis (partially discontinued): functionality from the `bayesflow.sensitivity` module. This is still
    possible, but we do no longer offer a special module for it. We plan to add a tutorial on this, see [#455](https://github.com/bayesflow-org/bayesflow/issues/455).
  * MCMC (discontinued): The `bayesflow.mcmc` module. We are considering other options
    to enable the use of BayesFlow in an MCMC setting.
  * Networks: `EvidentialNetwork`.
  * Model misspecification detection: MMD test in the summary space (see #384).

If you encounter any functionality that is missing and not listed here, please let us
know by opening an issue.

-------------

**Question:**
I still need the old BayesFlow for some of my projects. How can I install it?

**Answer:**
You can find and install the old Bayesflow version via the `stable-legacy` branch on GitHub.
The corresponding [documentation](https://bayesflow.org/stable-legacy/index.html) can be
accessed by selecting the "stable-legacy" entry in the version picker of the documentation.

You can also install the latest version of BayesFlow v1.x from PyPI using

```
pip install "bayesflow<2.0"
```

-------------

## Awesome Amortized Inference

If you are interested in a curated list of resources, including reviews, software, papers, and other resources related to amortized inference, feel free to explore our [community-driven list](https://github.com/bayesflow-org/awesome-amortized-inference). If you'd like a paper (by yourself or someone else) featured, please add it to the list with a pull request, an issue, or a message to the maintainers.

## Acknowledgments

This project is currently managed by researchers from Rensselaer Polytechnic Institute, TU Dortmund University, and Heidelberg University. It is partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) Projects 528702768 and 508399956. The project is further supported by Germany's Excellence Strategy -- EXC-2075 - 390740016 (Stuttgart Cluster of Excellence SimTech) and EXC-2181 - 390900948 (Heidelberg Cluster of Excellence STRUCTURES), the collaborative research cluster TRR 391 – 520388526, as well as the Informatics for Life initiative funded by the Klaus Tschira Foundation.

BayesFlow is a [NumFOCUS Affiliated Project](https://numfocus.org/sponsored-projects/affiliated-projects).
