# Introduction

*Disclaimer: This guide is in an early stage. We welcome contributions to the guide in form of issues and pull requests.*

Welcome to the User Guide! This guide is still in a very early stage, but we plan to evolve it into a comprehensive guide to using BayesFlow.

## Why (and When) Do We Need Amortized Bayesian Inference (ABI)?

In traditional Bayesian inference, we seek to approximate the posterior distribution of model parameters given observed data for each new data instance separately. This process can be computationally expensive, especially for complex models or large datasets, because it often involves iterative optimization or sampling methods. This step needs to be repeated for each new instance of data.

Amortized Bayesian inference offers a solution to this problem. “Amortization” here refers to spreading out the computational cost over multiple instances. Instead of computing a new posterior from scratch for each data instance, amortized inference learns a function. This function is parameterized by a neural network, that directly maps observations to an approximation of the posterior distribution. This function is trained over the dataset to approximate the posterior for *any* new data instance efficiently. In this example, we will use a simple Gaussian model to illustrate the basic concepts of amortized posterior estimation.

At a high level, our architecture consists of a summary network $\mathbf{h}$ and an inference network $\mathbf{f}$ which jointly learn to invert a generative model. The summary network transforms input data $\mathbf{x}$ of potentially variable size to a fixed-length representations. The inference network generates random draws from an approximate posterior $\mathbf{q}$ via a conditional generative networks (here, an invertible network).

## BayesFlow

BayesFlow offers flexible modules you can adapt to different Amortized Bayesian Inference (ABI) workflows. In brief:

* The module {py:mod}`~bayesflow.simulators` contains high-level wrappers for gluing together priors, simulators, and meta-functions, and generating all quantities of interest for a modeling scenario.
* The module {py:mod}`~bayesflow.adapters` contains utilities that preprocess the generated data from the simulator to a format more friendly for the neural approximators.
* The module {py:mod}`~bayesflow.networks` contains the core neural architecture used for various tasks, e.g., a generative {py:class}`~bayesflow.networks.FlowMatching` architecture for approximating distributions, or a {py:class}`~bayesflow.networks.DeepSet` for learning permutation-invariant summary representations (embeddings).
* The module {py:mod}`~bayesflow.approximators` contains high-level wrappers which connect the various networks together and instruct them about their particular goals in the inference pipeline.
