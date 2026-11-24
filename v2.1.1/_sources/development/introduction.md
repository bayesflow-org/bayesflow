# Introduction

From version 2 on, BayesFlow is built on [Keras3](https://keras.io/), which
allows writing machine learning pipelines that run in JAX, TensorFlow and PyTorch.
By using functionality provided by Keras, and extending it with backend-specific
code where necessary, we aim to build BayesFlow in a backend-agnostic fashion as
well.

As Keras is built upon three different backends, each with different functionality
and design decisions, it comes with its own quirks and compromises. The following documents
outline some of them, along with the design decisions and programming patterns
we use to counter them.
