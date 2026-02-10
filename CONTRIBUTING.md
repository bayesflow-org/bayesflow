# Contributing to BayesFlow

BayesFlow is a highly open source project. We welcome contributions from the community.

## How to contribute

First, take some time to read this whole guide.
Then, follow these steps to submit a contribution to BayesFlow:

### 1. Open an issue

Before you start any work, please open an issue on GitHub if one does not already exist. Describe the motivation for
the changes and add possible interfaces or use cases. We will determine the feature scope and discuss implementation
details with you.

Here is an example of a good issue:

------------------------------------------------------------------------------------------------------------------------

### #123: Add support for multi-level priors

#### Motivation:

There is currently a significant scientific push to use multi-level prior distributions for hierarchical models,
for instance in [1] and [2]. This feature would allow users to research these models more easily.

#### Possible Interface:

The multi-level graph structure could be given implicitly via the argument names of each sampling function.
For instance:

```python3
import bayesflow as bf
import keras


@bf.distribution
def prior1():
    return dict(a=keras.random.normal(), b=keras.random.normal())

@bf.distribution
def prior2(a):
    return dict(c=keras.random.normal(a))

@bf.distribution
def prior3(a, b, c):
    return dict(d=keras.random.normal(a + b + c))
```

#### References:

[1]: Some paper
[2]: Some other paper

------------------------------------------------------------------------------------------------------------------------

### 2. Set up your development environment

Once we agree on an approach in the issue you opened, we can move ahead with the implementation.

Most contributors will have to create and clone a fork of BayesFlow using the GitHub interface or the CLI:

```bash
gh repo fork bayesflow-org/bayesflow --clone
cd bayesflow
git checkout dev
```

Then, create a development environment with uv, or any other environment manager of your choice.
We recommend installing the optional `all` dependencies, which include packages for testing and documentation generation.
Don't forget to also install `pre-commit` hooks to ensure code quality and consistency.
**Your PR will likely not pass checks if you skip this step.**

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra all
pre-commit install
```

Note: always use a clean environment dedicated for development of BayesFlow to avoid dependency issues.

Finally, install at least one backend of your choice.
At the moment of writing this, to install all three backends with CUDA support, run:

```bash
uv pip install -U jax[cuda12] torch torchvision tensorflow[and-cuda]
```

Note that if you install multiple backends, BayesFlow will try to pick them according to the order of preference
`jax > torch > tensorflow` when you import it. To test your code under a specific backend, set the environment variable
`KERAS_BACKEND` to either `jax`, `torch`, or `tensorflow` before importing BayesFlow:

```python
import os
os.environ["KERAS_BACKEND"] = "torch"  # or jax, or tensorflow
import bayesflow as bf
```

You can also set the environment variable in your terminal session before running your code:

```bash
KERAS_BACKEND=torch python my_script.py
```

or export it for the whole session:

```bash
export KERAS_BACKEND=tensorflow  # or jax, or torch
```

You can also add this line to your `.bashrc` file (or equivalent) to make it permanent,
but be aware that this will affect all your BayesFlow code until you change it again.

### 3. Implement your changes

In general, we recommend a test-driven development approach:

1. Write a test for the functionality you want to add.
2. Write the code to make the test pass.

You can run tests for your installed environment using `pytest`:

```bash
pytest
```

Make sure to occasionally also run multi-backend tests for your OS using [tox](https://tox.readthedocs.io/en/latest/):

```bash
tox --parallel auto
```

See `tox.ini` for details on the environment configurations.
Multi-OS tests will automatically be run once you create a pull request.

Note that to be backend-agnostic, your code must not:
1. Use code from a specific machine learning backend
2. Use code from the `keras.backend` module
3. Rely on the specific tensor object type or semantics

Examples of bad code:
```py3
# bad: do not use specific backends
import tensorflow as tf
x = tf.zeros(3)

# bad: do not use keras.backend
shape = keras.backend.shape(x)  # will error under torch backend

# bad: do not use tensor methods directly
z = x.numpy()  # will error under torch backend if device is cuda
```

Use instead:
```py3
# good: use keras instead of specific backends
import keras
x = keras.ops.zeros(3)

# good: use keras.ops, keras.random, etc.
shape = keras.ops.shape(x)

# good: use keras methods instead of direct tensor methods
z = keras.ops.convert_to_numpy(x)
```

### 4. Document your changes

The documentation uses [sphinx](https://www.sphinx-doc.org/) and relies on [numpy style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html) in classes and functions.

The overall *structure* of the documentation is manually designed, but the API documentation is auto-generated.
New top-level modules (i.e., `bayesflow.mynewmodule`) have to be manually added to the list in `docsrc/source/api/bayesflow.rst` to be included.

You can re-build the for your local state with:

```bash
cd docsrc
make clean && make local-docs
# in case of issues, try `make clean-all`
```

Note that files ignored by git (i.e., listed in `.gitignore`) are not included in the documentation.

We also provide a multi-version documentation, which renders the branches `main` and `stable-legacy`. To generate it, run

```bash
cd docsrc
make clean && make production-docs
```

This will create and cache virtual environments for the build at `docsrc/.docs_venvs`.
To remove them, run `make clean-all` in the `docsrc` directory.

The entry point of the rendered documentation will be at `docs/index.html`.
To view the docs in the browser (which ensures correct redirects), run:

```bash
cd docsrc
make view-docs
```

See `docsrc/README.md` for more details.

Note that undocumented changes will likely be rejected.

### 5. Create a pull request

Once your changes are ready, create a pull request to the `dev` branch using the GitHub web interface, CLI, or tool of your choice.
Make sure to reference the issue you opened in step 1. If no issue exists, either open one first or follow the
issue guidelines for the pull request description.

Here is an example of a good pull request:

------------------------------------------------------------------------------------------------------------------------

### #124: Add support for multi-level priors

Resolves #123.

Multi-level priors are implemented via a graph structure which is internally created from the
argument names of the sampling functions.

------------------------------------------------------------------------------------------------------------------------

## Tutorial Notebooks

New tutorial notebooks are always welcome! You can add your tutorial notebook file to `examples/` and add a reference
to the list of notebooks in `docsrc/source/examples.rst`.
Re-build the documentation (see above) and your notebook will be included.
