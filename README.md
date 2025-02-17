# BayesFlow <img src="img/bayesflow_hex.png" style="float: right; width: 20%; height: 20%;" align="right" alt="BayesFlow Logo" />
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/bayesflow-org/bayesflow/tests.yaml?style=for-the-badge&label=Tests)
![Codecov](https://img.shields.io/codecov/c/github/bayesflow-org/bayesflow/dev?style=for-the-badge)
[![DOI](https://img.shields.io/badge/DOI-10.21105%2Fjoss.05702-blue?style=for-the-badge)](https://doi.org/10.21105/joss.05702)
![PyPI - License](https://img.shields.io/pypi/l/bayesflow?style=for-the-badge)

BayesFlow is a Python library for simulation-based **Amortized Bayesian Inference** with neural networks.
It provides users with:

- A user-friendly API for rapid Bayesian workflows
- A rich collection of neural network architectures
- Multi-Backend Support via [Keras3](https://keras.io/keras_3/): You can use [PyTorch](https://github.com/pytorch/pytorch), [TensorFlow](https://github.com/tensorflow/tensorflow), or [JAX](https://github.com/google/jax)

BayesFlow is designed to be a flexible and efficient tool that enables rapid statistical inference
fueled by continuous progress in generative AI and Bayesian inference.

## Conceptual Overview

<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./img/bayesflow_landing_dark.jpg">
  <source media="(prefers-color-scheme: light)" srcset="./img/bayesflow_landing_light.jpg">
  <img alt="dsd" src="./img/bayesflow_landing_dark.jpg">
</picture>
</div>

A cornerstone idea of amortized Bayesian inference is to employ generative
neural networks for parameter estimation, model comparison, and model validation
when working with intractable simulators whose behavior as a whole is too
complex to be described analytically.

## Disclaimer

This is the current dev version of BayesFlow, which constitutes a complete refactor of the library built on Keras 3. This way, you can now use any of the major deep learning libraries as backend for BayesFlow. The refactor is still work in progress with some of the advanced features not yet implemented. We are actively working on them and promise to catch up soon.

If you encounter any issues, please don't hesitate to open an issue here on [Github](https://github.com/bayesflow-org/bayesflow/issues) or ask questions on our [Discourse Forums](https://discuss.bayesflow.org/).

## Install

### Backend

First, install one machine learning backend of choice. Note that BayesFlow **will not run** without a backend.

- [Install JAX](https://jax.readthedocs.io/en/latest/installation.html)
- [Install PyTorch](https://pytorch.org/get-started/locally/)
- [Install TensorFlow](https://www.tensorflow.org/install)

If you don't know which backend to use, we recommend JAX to get started.
It is the fastest backend and already works pretty reliably with the current
dev version of bayesflow.

Once installed, [set the backend environment variable as required by keras](https://keras.io/getting_started/#configuring-your-backend). For example, inside your Python script write:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import bayesflow
```

If you use conda, you can alternatively set this individually for each environment in your terminal. For example:

```bash
conda env config vars set KERAS_BACKEND=jax
```

This way, you also don't have to manually set the backend every time you are starting Python to use BayesFlow.

**Caution:** Some people report that the IDE (e.g., VSCode or PyCharm) can silently overwrite environment variables. If you have set your backend as an environment variable and you still get keras-related import errors when loading BayesFlow, these IDE shenanigans might be the culprit. Try setting the keras backend in your Python script via `import os; os.environ["KERAS_BACKEND"] = "<YOUR-BACKEND>"`.

### Using pip

You can install the dev version with pip:

```bash
pip install git+https://github.com/bayesflow-org/bayesflow@dev
```

### Using Conda (coming soon)

The dev version is not conda-installable yet.

### From Source

If you want to contribute to BayesFlow, we recommend installing the dev branch from source:

```bash
git clone -b dev git@github.com:bayesflow-org/bayesflow.git
cd <local-path-to-bayesflow-repository>
conda env create --file environment.yaml --name bayesflow
```

## Getting Started

Check out some of our walk-through notebooks below. We are actively working on porting all notebooks to the new interface so more will be available soon!

1. [Linear regression starter example](examples/Linear_Regression_Starter.ipynb)
2. [Two moons starter example](examples/Two_Moons_Starter.ipynb)
3. [SIR model with custom summary network](examples/SIR_Posterior_Estimation.ipynb)
4. [SBML model using an external simulator](examples/SBML_Posterior_Estimation.ipynb)
5. [Hyperparameter optimization](examples/Hyperparameter_Optimization.ipynb)
6. [Bayesian experimental design](examples/Bayesian_Experimental_Design.ipynb)
7. More coming soon...

## Documentation \& Help

Documentation is available at https://bayesflow.org. Please use the [BayesFlow Forums](https://discuss.bayesflow.org/) for any BayesFlow-related questions and discussions, and [GitHub Issues](https://github.com/bayesflow-org/bayesflow/issues) for bug reports and feature requests.

## Citing BayesFlow

You can cite BayesFlow along the lines of:

- We approximated the posterior with neural posterior estimation and learned summary statistics (NPE; Radev et al., 2020), as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023a).
- We approximated the likelihood with neural likelihood estimation (NLE; Papamakarios et al., 2019) without hand-crafted summary statistics, as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).
- We performed simultaneous posterior and likelihood estimation with jointly amortized neural approximation (JANA; Radev et al., 2023a), as implemented in the BayesFlow software for amortized Bayesian workflows (Radev et al., 2023b).

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

## Acknowledgments

This project is currently managed by researchers from Rensselaer Polytechnic Institute, TU Dortmund University, and Heidelberg University. It is partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation, Project 528702768). The project is further supported by Germany's Excellence Strategy -- EXC-2075 - 390740016 (Stuttgart Cluster of Excellence SimTech) and EXC-2181 - 390900948 (Heidelberg Cluster of Excellence STRUCTURES), as well as the Informatics for Life initiative funded by the Klaus Tschira Foundation.
