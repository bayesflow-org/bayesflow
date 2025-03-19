# Documentation

## Overview

To install the necessary dependencies, please run `pip install -e .[docs]` in the root directory.
You can then do the following (inside the `docsrc` directory):

1. `make local`: Generate the docs for the current local state
2. `make docs`: Build the docs for branches and tags specified in `poly.py` in sequential fashion. Virtual environments are cached (run `make clean-all` to delete)
3. `make docs-sequential`: As `make docs`, but versions are built sequentially, and the build environment is deleted after each build (see below for details)
4. `make view-docs`: Starts a local webserver to display the content of `../docs`

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

Please refer to the [sphinx-polyversion documentation](https://real-yfprojects.github.io/sphinx-polyversion/1.1.0/index.html)
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
- `source/api/bayesflow.rst`: Specify the submodules that should be included in the docs.

### Building

For the multi-version docs, there are two ways to build them, which can be
configured by setting the `--sequential` flag.

#### Parallel Builds (Default)

This is the faster, but more resource intensive way. All builds run in parallel,
in different virtual environments which are cached between runs.
Therefore it needs a lot of space (around 20GB), some memory, and the runtime
is determined by the slowest build.

#### Sequential Builds

By setting the `--sequential` flag in the `sphinx-polyversion` call, a
resource-constrained approach is chosen. Builds are sequential, and the
virtual environment is deleted after the build. This overcomes the disk space
limitations in the GitHub actions, at the cost of slightly higher built times.
This is used in `make docs-sequential`.

### Internals

We extend `sphinx-polyversion` in minor ways, for some fixes there are already
open PRs in the upstream project. The patched classes are located in
`polyversion_patches.py`.

We customize the creation and loading of the virtual environment to have
one environment per revision (`DynamicPip`). Setting `temporary` creates the
environment in the temporary build directory, so that it will be removed after
the build.

As only the contents of a revision, but not the `.git` folder is copied
for the build, we have to supply `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_BAYESFLOW`
with a version, otherwise `setuptools-scm` will fail when running
`pip install -e .`.

For all other details, please refer to `poly.py` and the code of `sphinx-polyversion`.

This text was written by @vpratz, if you have any questions feel free to reach out.
