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

Create a fork of the BayesFlow git repository using the GitHub interface.
Then clone your fork and install the development environment with conda:

```bash
git clone https://github.com/<your-username>/bayesflow
cd bayesflow
git checkout dev
conda env create --file environment.yaml --name bayesflow
conda activate bayesflow
pre-commit install
```

We recommend using the PyTorch backend for development.
Be careful not to downgrade your keras version when installing the backend.

You can also install a triple backend cuda environment using `cuda.yaml`:

```bash
conda env create --file cuda.yaml --name bf-cuda
```

Note that we cannot guarantee that this environment file will always work,
as it is not tested frequently or across devices.
It is provided merely as a convenience. For reliability, install only a single backend.

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

Run the following command to install all necessary packages for setting up documentation generation:

```
pip install .[docs]
```

The overall *structure* of the documentation is manually designed, but the API documentation is auto-generated.

You can re-build the current documentation with

```bash
cd docsrc
make clean && make dev
# in case of issues, try `make clean-all`
```

We also provide a multi-version documentation. To generate it, run

```bash
cd docsrc
make clean && make github
```

The entry point of the rendered documentation will be at `docs/index.html`.

Note that undocumented changes will likely be rejected.

### 5. Create a pull request

Once your changes are ready, create a pull request to the `dev` branch using the GitHub interface.
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
