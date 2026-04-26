# Introduction

Bayesian inference gives us a principled language for learning from data under uncertainty. We write down a generative model, combine prior knowledge with observations, and obtain a posterior distribution over the quantities we care about. This posterior is often the real object of interest: it tells us not only *what values are plausible*, but also *how (un)certain we should be*.

The challenge is computation.

For many modern models, posterior inference is slow or unavailable with standard methods. The likelihood may be impossible to evaluate. The simulator may be stochastic, high-dimensional, ill-defined, or expensive. The data may have variable size or complex structure. And even when conventional Markov chain Monte Carlo or variational methods work, they often need to be rerun from scratch for every new dataset, every new participant, every new experimental design, or every new observation.

Amortized Bayesian inference addresses this bottleneck by changing the unit of computation. Instead of solving a new inference problem from scratch every time we observe new data, we first invest in learning an inference machine. We simulate many datasets from a generative model and train a neural network to map simulated observations to the corresponding Bayesian quantities of interest: posterior samples, likelihoods, likelihood ratios, Bayes estimators, point estimates, model probabilities, or any distribution we may care about.

After training, inference becomes fast.

This is the core idea of amortization: we pay an upfront simulation-and-training cost once, and then reuse the learned estimator for many future inference tasks from the same model family. In settings where we want to test models on simulated data, data arrive repeatedly, experiments are run sequentially, models are compared many times, or inference must be performed interactively, this changes Bayesian inference from a slow per-dataset computation into a reusable workflow.

## Amortized Bayesian Workflows

At a high level, an amortized Bayesian workflow has three ingredients:

1. **A generative model** that can simulate parameters, latent variables, observations, metadata or other quantities of interest.
2. **A data representation pipeline** that converts raw simulator output into the structured tensors required by neural networks.
3. **A neural inference engine** that learns the inverse mapping from observations back to uncertainty-aware Bayesian conclusions.

BayesFlow is the workflow layer that connects these pieces.

It lets you define the generative process, adapt simulator output into training-ready form, choose from modern neural architectures for the inference task, train the estimator, and evaluate its fidelity with simulation-based diagnostics. The goal is not merely to fit a neural network. The goal is to build a reusable Bayesian inference workflow that scales to complex simulators and produces uncertainty-aware outputs quickly once trained.

This guide introduces all components that let you create and run amortized workflows. These component are organized around several modules:

{py:mod}~bayesflow.simulators provides tools for defining and combining priors, simulators, and meta-functions. These components generate the model-implied quantities used for training, validation, diagnostics, and inference.
{py:mod}~bayesflow.adapters defines the bridge between simulator output and neural-network input. Adapters make preprocessing explicit, reproducible, and shared between training and inference.
{py:mod}~bayesflow.networks contains neural architectures for amortized inference and representation learning, including generative networks for posterior approximation and summary networks for structured or variable-size observations.
{py:mod}~bayesflow.approximators connects networks to concrete inference goals, such as posterior estimation, likelihood estimation, ratio estimation, or point estimation.
High-level workflows, such as {py:class}~bayesflow.workflows.BasicWorkflow, orchestrate the full process from simulation to training, diagnostics, and application.

## Why BayesFlow?

BayesFlow is useful because it gives you a complete, flexible workflow for amortized Bayesian inference:

* **One workflow from simulation to inference.**
  BayesFlow connects the full pipeline: simulate data, adapt simulator outputs, train neural approximators, diagnoze the results, and apply the trained workflow to new data.

* **Multi-backend support.**
  BayesFlow runs on the Keras 3 ecosystem, so the same workflow can use JAX, PyTorch, or TensorFlow backends depending on your hardware, speed requirements, and existing ML stack.

* **Interchangeable components.**
  Every major part of the workflow is modular. You can swap the simulator, adapter, summary network, inference network, inference tragets, or training strategy without rewriting the entire project.

* **Reusable inference after training.**
  Once trained, the amortized estimator can produce fast posterior samples, point estimates, likelihoods, ratios, model probabilities, or other model-implied quantities for new data.

* **Explicit preprocessing with adapters.**
  BayesFlow makes preprocessing part of the workflow. Adapters define how raw simulator outputs become neural-network-ready inputs, helping keep training and inference consistent.

* **Built-in diagnostics.**
  BayesFlow includes many diagnostics for checking whether the approximation is trustworthy, including calibration, parameter recovery, posterior contraction, and posterior predictive checks.

* **High-level workflows, low-level flexibility.**
  You can use high-level workflow objects for convenience, while still accessing lower-level components compatible with the entire deep learning ecosystem when you need more control.

* **Agentic AI-ready by design.**
  Because BayesFlow workflows are explicit, modular, and diagnostic-driven, they are well suited for AI agents that need to reason about simulator design, adapter choices, neural architectures, training modes, and validation checks. See the [Baygent amortized-workflow skill](https://github.com/Learning-Bayesian-Statistics/baygent-skills/tree/main) for an example.
