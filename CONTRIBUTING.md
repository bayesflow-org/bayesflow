# Contributing to BayesFlow

BayesFlow is an open-source project. We welcome contributions from the community.

## How to contribute

First, take some time to read this whole guide. Then, follow these steps to submit a contribution to BayesFlow:

### 1. Discuss the change

Before starting substantial work, search [GitHub Issues](https://github.com/bayesflow-org/bayesflow/issues) and open an issue if one does not already exist. Describe the motivation, expected behavior, possible interfaces or use cases, and any relevant references. This lets us agree on the scope and implementation approach before work begins.

Small fixes, such as typo or documentation corrections, may be submitted directly as a pull request. Explain their motivation and scope in the pull request description.

### 2. Set up your development environment

Once the scope is clear, you can move ahead with the implementation. Use a clean environment dedicated to BayesFlow development to avoid dependency issues. BayesFlow supports Python 3.12 and 3.13; the examples below use Python 3.12 to match the CI baseline.

Most contributors will need to create and clone a fork of BayesFlow. You can use the GitHub interface or the CLI:

```bash
gh repo fork bayesflow-org/bayesflow --clone
cd bayesflow
git switch dev
```

We recommend [uv](https://docs.astral.sh/uv/), which is also used by our CI. After [installing uv](https://docs.astral.sh/uv/getting-started/installation/), create the project environment and install all development dependencies:

```bash
uv sync --python 3.12 --extra all
uv run pre-commit install
```

If you prefer conda and pip:

```bash
conda create -n bf python=3.12
conda activate bf
python -m pip install -e ".[all]"
pre-commit install
```

Finally, install at least one Keras backend. We recommend JAX for development; with uv, install its CPU version using:

```bash
uv pip install -U jax
```

For GPU installations or other backends, follow the official installation guides for [JAX](https://docs.jax.dev/en/latest/installation.html), [PyTorch](https://pytorch.org/get-started/locally/), or [TensorFlow](https://www.tensorflow.org/install). The repository's [`cuda.yaml`](https://github.com/bayesflow-org/bayesflow/blob/dev/cuda.yaml) provides a maintained example for an all-backend NVIDIA environment, but you may need to adapt it to your driver and CUDA versions.

Set the backend environment variable before importing Keras or BayesFlow. We recommend defaulting to JAX:

```bash
# In the current shell or your shell profile
export KERAS_BACKEND=jax
```

If you use conda, you can set this per-environment:

```bash
conda env config vars set KERAS_BACKEND=jax
```

### 3. Implement your changes

In general, we recommend a test-driven development approach:

1. Write a test for the functionality you want to add.
2. Write the code to make the test pass.

You can run the regular test suite for your installed environment using `pytest`:

```bash
uv run pytest -m "not slow"
```

Slow integration tests, including notebook execution tests, may require example-specific dependencies. Run the relevant slow tests separately when your changes affect them.

The default pytest configuration excludes `tests/test_experimental`. If your changes affect experimental features, run those tests explicitly:

```bash
uv run pytest -m "not slow" --override-ini="addopts=--cov=bayesflow --strict-markers --verbose" tests/test_experimental
```

Run multi-backend tests for your operating system using [tox](https://tox.wiki/):

```bash
uv run tox --parallel auto
```

See [`tox.ini`](https://github.com/bayesflow-org/bayesflow/blob/dev/tox.ini) for details on the environment configurations.
Multi-OS tests will automatically be run once you create a pull request.

Before opening a pull request, run all configured style checks:

```bash
uv run pre-commit run --all-files
```

Note that to be backend-agnostic, your code must not:

1. Use code from a specific machine learning backend
2. Use code from the `keras.backend` module
3. Rely on the specific tensor object type or semantics

Examples of bad code:

```python
# bad: do not use specific backends
import tensorflow as tf
x = tf.zeros(3)

# bad: do not use keras.backend
shape = keras.backend.shape(x)  # not part of the Keras 3 backend API

# bad: do not use tensor methods directly
z = x.numpy()  # will error under the PyTorch backend if the device uses CUDA
```

Use instead:

```python
# good: use keras instead of specific backends
import keras
x = keras.ops.zeros(3)

# good: use keras.ops, keras.random, etc.
shape = keras.ops.shape(x)

# good: use keras methods instead of direct tensor methods
z = keras.ops.convert_to_numpy(x)
```

### 4. Document your changes

The documentation uses [Sphinx](https://www.sphinx-doc.org/) and relies on [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) in classes and functions.

The `all` extra installed during setup includes the documentation dependencies. If you started from a narrower uv environment, install the `docs` extra without removing a separately installed backend:

```bash
uv sync --extra docs --inexact
```

The overall *structure* of the documentation is manually designed, but the API documentation is auto-generated.
New top-level modules (i.e., `bayesflow.mynewmodule`) have to be manually added to [`docsrc/source/api/bayesflow.rst`](https://github.com/bayesflow-org/bayesflow/blob/dev/docsrc/source/api/bayesflow.rst) to be included.

You can rebuild the documentation for your local checkout with:

```bash
cd docsrc
uv run make clean && uv run make local-docs
# in case of issues, try `uv run make clean-all`
```

Untracked files excluded by `.gitignore` are not copied into the documentation build.

Most contributors only need the local documentation build. The multi-version build renders the `stable-legacy` branch and selected release tags configured in [`docsrc/poly.py`](https://github.com/bayesflow-org/bayesflow/blob/dev/docsrc/poly.py). It requires the complete Git history and tags, a local `stable-legacy` branch, and network access for creating per-version environments. To generate it, run:

```bash
cd docsrc
uv run make clean && uv run make production-docs
```

This parallel build creates and caches virtual environments at `docsrc/.docs_venvs`. On resource-constrained systems, use `uv run make production-docs-sequential` instead. To remove cached environments, run `uv run make clean-all` in the `docsrc` directory.

The entry point of the rendered documentation will be at `docs/index.html`.
To view the docs in the browser (which ensures correct redirects), run:

```bash
cd docsrc
uv run make view-docs
```

See [`docsrc/README.md`](https://github.com/bayesflow-org/bayesflow/blob/dev/docsrc/README.md) for more details.

Note that undocumented changes will likely be rejected.

### 5. Create a pull request

Once your changes are ready, create a pull request to the `dev` branch using the GitHub interface.
Reference the related issue when one exists. For a small change submitted without an issue, include the same motivation and context in the pull request description. Summarize the implementation and list the tests and documentation you added or updated.

## Tutorial Notebooks

New public tutorial notebooks are always welcome! Add the notebook directly under `examples/`, update the tutorial lists in [`README.md`](https://github.com/bayesflow-org/bayesflow/blob/dev/README.md) and [`docsrc/source/index.md`](https://github.com/bayesflow-org/bayesflow/blob/dev/docsrc/source/index.md), and add its toctree entry to [`docsrc/source/examples.rst`](https://github.com/bayesflow-org/bayesflow/blob/dev/docsrc/source/examples.rst). Add an execution test under `tests/test_examples/` when practical, then rebuild the documentation as described above.

Notebooks that should not appear in the public documentation belong under `examples/misc/`; supporting assets may also live there.
