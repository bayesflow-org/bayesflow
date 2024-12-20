# Documentation

## Overview

To install the necessary dependencies, please run `pip install -e .[docs]`.
You can then do the following:

1. `make dev`: Generate the docs for the current version
2. `make github`: Build the docs for tagged versions, `master` and `dev` in a sequential fashion
3. `make parallel`: As `make github`, but builds occur in parallel (see below for details)

The docs will be copied to `../docs`.

## Build process

In this section, the goals and constraints for the build process are described.

Goals:

- (semi-)automated documentation generation
- multi-version documentation
- runnable as a GitHub action

Constraints:

- GitHub actions have limited disk space (14GB)

### Considerations

For building the documentation, we need to install a given BayesFlow
version/branch, its dependencies and the documentation dependencies into
a virtual environment. As the dependencies differ, we cannot share the
environments between versions.

[sphinx-polyversion](https://github.com/real-yfprojects/sphinx-polyversion/) is a compact standalone tool that handles this case in a customizable manner.

### Setup

Please refer to the [sphinx-polyversion documentation](https://real-yfprojects.github.io/sphinx-polyversion/1.0.0/index.html)
for a getting started tutorial and general documentation.
Important locations are the following:

- `poly.py`: Contains the polyversion-specific configuration.
- `pre-build.py`: Build script to move files from other locations to `source`.
    Shared between all versions.
- `source/conf.py`: Contains the sphinx-specific configuration. Will be copied
    from the currently checked-out branch, and shared between all versions.
    This enables a unified look and avoids having to add commits to old versions.
- `polyversion/`: Polyversion-specific files, currently only redirect template.
- `Makefile`/`make.bat`: Define commands to build different configurations.

### Building

For the multi-version docs, there are two ways to build them, which can be
configured by setting the `BF_DOCS_SEQUENTIAL_BUILDS` environment variable.

#### Parallel Builds (Default)

This is the faster, but more resource intensive way. All builds run in parallel,
in different virtual environments which are cached between runs.
Therefore it needs a lot of space (around 20GB), some memory, and the runtime
is determined by the slowest build.

#### Sequential Builds

By setting the environment variable `BF_DOCS_SEQUENTIAL_BUILDS=1`, a
resource-constrained approach is chosen. Builds are sequential, and the
virtual environment is deleted after the build. This overcomes the disk space
limitations in the GitHub actions, at the cost of slightly higher built times
(currently about 30 minutes). The variable can be set in the following way,
which is used in `make github`:

```bash
BF_DOCS_SEQUENTIAL_BUILDS=1 sphinx-polyversion -vv poly.py
```

### Internals

We customize the creation and loading of the virtual environment to have
one environment per revision (`DynamicPip`). We also create a variant that
removes the environment after leaving it (`DestructingDynamicPip`). This
enables freeing disk space in sequential builds.

As only the contents of a revision, but not the `.git` folder is copied
for the build, we have to supply `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BAYESFLOW`
with a version, otherwise `setuptools-scm` will fail when running
`pip install -e .`. To enable this, we accept and set environment variables
in `DynamicPip`.

The environments are created in `VENV_DIR_NAME`, and only removed if they are
in this directory.

For sequential builds, define the `SynchronousDriver` class, which builds the
revisions sequentially.

To generate redirects for the old, non-version docs, we need the driver to
create folders for the rendered templates. This extension takes place in
`TemplatingDriver`.

For all other details, please refer to `poly.py` and the code of `sphinx-polyversion`.

This text was written by @vpratz, if you have any questions feel free to reach out.
